import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
DEFAULT_MAX_TURNS = int(os.getenv("DEEPDIVE_MAX_TURNS", "64"))
DEFAULT_TOOL_SERVER_URL = os.getenv("DEEPDIVE_TOOL_SERVER_URL", "http://127.0.0.1:7230/tool")
DEFAULT_REWARD_SERVER_URL = os.getenv("DEEPDIVE_REWARD_SERVER_URL", "http://127.0.0.1:8888/evaluate")
RETURN_LOGPROB = os.getenv("DEEPDIVE_RETURN_LOGPROB", "1").lower() in {"1", "true", "yes"}


def _get_arg(args, name: str, default=None):
    return getattr(args, name, default)


def _render_prompt(
    state: GenerateState,
    prompt: Union[str, List[Dict[str, Any]]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if isinstance(prompt, str):
        return prompt
    try:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        return state.tokenizer.apply_chat_template(prompt, **kwargs)
    except Exception:
        lines = []
        for item in prompt:
            role = item.get("role", "user").upper()
            lines.append(f"{role}: {item.get('content', '')}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)


def _extract_tool_call(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if TOOL_CALL_START not in text:
        return None, "missing_tool_call"
    tool_part = text.rsplit(TOOL_CALL_START, 1)[-1]
    if TOOL_CALL_END in tool_part:
        tool_part = tool_part.split(TOOL_CALL_END, 1)[0]
    tool_part = tool_part.strip()
    try:
        tool_call = json.loads(tool_part)
    except Exception:
        return None, "malformed_tool_call"
    if not isinstance(tool_call, dict):
        return None, "malformed_tool_call"
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if name not in {"search", "click", "open", "finish"} or not isinstance(arguments, dict):
        return None, "invalid_tool_call"
    return tool_call, None


def _build_observation(tool_result: Dict[str, Any]) -> str:
    observation = tool_result.get("observation", tool_result.get("output", ""))
    if not isinstance(observation, str):
        observation = json.dumps(observation, ensure_ascii=False)
    return f"\n<observation>\n{observation}\n</observation>\n"


def _remote_env_info(sample: Sample) -> Dict[str, Any]:
    metadata = getattr(sample, "metadata", {}) or {}
    if "remote_env_info" in metadata:
        return metadata["remote_env_info"] or {}
    prompt = getattr(sample, "prompt", "")
    if isinstance(prompt, dict):
        return prompt.get("remote_env_info", {}) or {}
    return {}


async def _call_tool_server(
    tool_server_url: str,
    session_id: str,
    tool_call: Dict[str, Any],
    remote_env_info: Dict[str, Any],
    timeout: int,
    max_retry: int,
) -> Dict[str, Any]:
    payload = {
        "session_id": session_id,
        "name": tool_call["name"],
        "arguments": tool_call.get("arguments", {}) or {},
        "remote_env_info": remote_env_info,
    }
    last_error = None
    for attempt in range(max_retry):
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(tool_server_url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retry:
                await asyncio.sleep(1)
    return {"observation": f"Tool call failed: {last_error}"}


def _append_text(
    state: GenerateState,
    text: str,
    response_parts: List[str],
    response_token_ids: List[int],
    loss_mask: List[int],
    mask_value: int,
    rollout_log_probs: Optional[List[float]] = None,
) -> None:
    token_ids = state.tokenizer(text, add_special_tokens=False)["input_ids"]
    response_parts.append(text)
    response_token_ids.extend(token_ids)
    loss_mask.extend([mask_value] * len(token_ids))
    if rollout_log_probs is not None:
        rollout_log_probs.extend([0.0] * len(token_ids))


async def generate_with_tool(args, sample: Sample, sampling_params) -> Sample:
    """DeepDive multi-turn rollout strategy for slime.

    The policy generates a `<tool_call>...</tool_call>` block. Non-finish tool
    calls are executed by the DeepDive tool server and appended as observation
    text with `loss_mask=0`.
    """
    assert not args.partial_rollout, "DeepDive tool rollout does not support partial rollout yet."

    state = GenerateState(args)
    tools = (sample.metadata or {}).get("tools")
    prompt_text = _render_prompt(state, sample.prompt, tools=tools)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response_parts: List[str] = []
    response_token_ids: List[int] = []
    loss_mask: List[int] = []
    rollout_log_probs: Optional[List[float]] = [] if RETURN_LOGPROB else None

    tool_server_url = os.getenv("DEEPDIVE_TOOL_SERVER_URL", DEFAULT_TOOL_SERVER_URL)
    max_turns = int(os.getenv("DEEPDIVE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
    tool_timeout = int(os.getenv("DEEPDIVE_TOOL_TIMEOUT", "300"))
    tool_max_retry = int(os.getenv("DEEPDIVE_TOOL_MAX_RETRY", "5"))
    stop_once_illform = os.getenv("DEEPDIVE_STOP_ONCE_ILLFORM", "1").lower() in {"1", "true", "yes"}
    session_id = sample.session_id or f"deepdive-{sample.index}-{time.time_ns()}"
    remote_env_info = _remote_env_info(sample)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    stop_tags = [TOOL_CALL_END]
    existing_stop = sampling_params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    sampling_params = {**sampling_params, "stop": list(dict.fromkeys([*existing_stop, *stop_tags]))}

    output = None
    error_reason = None
    finished = False
    for _ in range(max_turns):
        payload = {
            "sampling_params": sampling_params,
            "input_ids": prompt_token_ids + response_token_ids,
        }
        if RETURN_LOGPROB:
            payload["return_logprob"] = True
        if getattr(args, "use_rollout_routing_replay", False):
            payload["return_routed_experts"] = True

        headers = None
        if session_id and getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": session_id}

        output = await post(url, payload, headers=headers)
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        if RETURN_LOGPROB:
            token_logprobs = output["meta_info"].get("output_token_logprobs")
            if token_logprobs is None:
                raise RuntimeError("output_token_logprobs missing from sglang response")
            cur_token_ids = [item[1] for item in token_logprobs]
            cur_log_probs = [item[0] for item in token_logprobs]
            response_parts.append(cur_response)
            response_token_ids.extend(cur_token_ids)
            loss_mask.extend([1] * len(cur_token_ids))
            rollout_log_probs.extend(cur_log_probs)
        else:
            _append_text(state, cur_response, response_parts, response_token_ids, loss_mask, 1)

        if finish_reason == "length":
            error_reason = "length"
            break

        tool_call, parse_error = _extract_tool_call(cur_response)
        if parse_error:
            error_reason = parse_error
            if stop_once_illform:
                break
            observation = f"\n<observation>\nInvalid tool call: {parse_error}. Use search, click, open, or finish.\n</observation>\n"
            _append_text(state, observation, response_parts, response_token_ids, loss_mask, 0, rollout_log_probs)
            continue

        if tool_call["name"] == "finish":
            error_reason = None
            finished = True
            break

        tool_result = await _call_tool_server(
            tool_server_url=tool_server_url,
            session_id=session_id,
            tool_call=tool_call,
            remote_env_info=remote_env_info,
            timeout=tool_timeout,
            max_retry=tool_max_retry,
        )
        observation = _build_observation(tool_result)
        _append_text(state, observation, response_parts, response_token_ids, loss_mask, 0, rollout_log_probs)

    if not finished and error_reason is None:
        error_reason = "max_turns"

    response = "".join(response_parts)
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = response
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    sample.session_id = session_id
    sample.metadata = {
        **(sample.metadata or {}),
        "deepdive_rollout_error": error_reason,
        "format_valid": error_reason is None,
    }
    if rollout_log_probs is not None:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    if output is None:
        sample.status = Sample.Status.FAILED
    else:
        sample.update_from_meta_info(args, output["meta_info"])
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
        elif finish_type == "abort":
            sample.status = Sample.Status.ABORTED
        elif finish_type == "stop":
            sample.status = Sample.Status.COMPLETED if error_reason is None else Sample.Status.FAILED
    return sample


generate = generate_with_tool


async def reward_func(args, sample: Sample, **kwargs):
    reward_server_url = os.getenv("DEEPDIVE_REWARD_SERVER_URL", DEFAULT_REWARD_SERVER_URL)
    payload = {
        "history": [
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": sample.response},
        ],
        "response": sample.response,
        "label": sample.label,
        "task_unfinished": sample.status != Sample.Status.COMPLETED,
        "format_valid": (sample.metadata or {}).get("format_valid", True),
        "remote_env_info": _remote_env_info(sample),
    }
    timeout = aiohttp.ClientTimeout(total=int(os.getenv("DEEPDIVE_REWARD_TIMEOUT", "600")))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(reward_server_url, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()
    return result.get("reward", 0.0)

import argparse
import json
import re
import traceback
from typing import Any, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI


JUDGE_PROMPT = """You are judging a DeepDive deep-search agent answer.

Question:
{question}

Reference answer:
{answer}

Agent response:
{response}

Decide whether the response answers the question correctly. Be strict about
entity identity, dates, and numeric values, but accept equivalent wording.

Return exactly:
Correct: yes/no
Extracted final answer: <short answer>
Reason: <one sentence>
"""


app = FastAPI()
reward_model = None


class DeepDiveJudge:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(proxy=None),
        )

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        last_error = None
        for _ in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
        print(f"[DeepDiveRewardServer] judge request failed: {last_error}")
        return ""


def _extract_response(history: List[Dict[str, Any]]) -> str:
    for item in reversed(history):
        if item.get("role") == "assistant":
            content = item.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def _extract_question(data: Dict[str, Any]) -> str:
    remote = data.get("remote_env_info", {}) or {}
    forbidden = remote.get("forbidden_texts") or []
    if forbidden:
        return str(forbidden[0])
    history = data.get("history") or []
    for item in history:
        if item.get("role") == "user":
            return str(item.get("content", ""))
    return str(data.get("question", ""))


def _parse_answer_correctness(text: str) -> Dict[str, Any]:
    correct_match = re.search(r"(?i)correct\s*:\s*(yes|no)", text)
    answer_match = re.search(r"(?i)extracted final answer\s*:\s*(.+)", text)
    correct = correct_match.group(1).lower() == "yes" if correct_match else False
    return {
        "reward": 1.0 if correct else 0.0,
        "judgement": text,
        "extracted_final_answer": answer_match.group(1).strip() if answer_match else "",
    }


def _format_is_valid(data: Dict[str, Any], response: str) -> bool:
    if data.get("task_unfinished", False):
        return False
    if data.get("format_valid") is False:
        return False
    if data.get("invalid_tool_call") or data.get("malformed_tool_call"):
        return False
    if not response.strip():
        return False
    return True


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        data = await request.json()
        label = data.get("label", "")
        if isinstance(label, list):
            label = label[0] if label else ""
        history = data.get("history", [])
        response = data.get("response") or _extract_response(history)
        question = _extract_question(data)

        format_reward = 1.0 if _format_is_valid(data, response) else 0.0
        if not question or not response or format_reward == 0.0:
            return {
                "reward": 0.0,
                "format_reward": format_reward,
                "answer_reward": 0.0,
                "judgement": "",
                "extracted_final_answer": "",
            }

        prompt = JUDGE_PROMPT.format(question=question, answer=label, response=response)
        judgement = await reward_model.complete([{"role": "user", "content": prompt}])
        parsed = _parse_answer_correctness(judgement)
        answer_reward = parsed["reward"]
        final_reward = 1.0 if format_reward == 1.0 and answer_reward == 1.0 else 0.0
        return {
            "reward": final_reward,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "judgement": parsed["judgement"],
            "extracted_final_answer": parsed["extracted_final_answer"],
        }
    except Exception as exc:
        print(f"[DeepDiveRewardServer] error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str, required=True)
    return parser.parse_args()


@app.on_event("startup")
async def startup_event():
    global reward_model
    args = get_args()
    reward_model = DeepDiveJudge(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    args = get_args()
    reward_model = DeepDiveJudge(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=False)

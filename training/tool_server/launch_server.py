import argparse
import asyncio
import time
from collections import defaultdict
from functools import wraps

from quart import Quart, jsonify, request

try:
    from .web_search import parse_url, search
except ImportError:
    from web_search import parse_url, search


parser = argparse.ArgumentParser()
parser.add_argument("--search_provider", type=str, choices=["serpapi", "serper"], default="serpapi")
parser.add_argument("--serp_api_key", type=str, default="")
parser.add_argument("--serper_api_key", type=str, default="")
parser.add_argument("--jina_api_key", type=str, default="")
parser.add_argument("--http_proxy", type=str, default=None)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=7230)
args = parser.parse_args()

app = Quart(__name__)
session2state = defaultdict(dict)


def log_tool_call(func):
    @wraps(func)
    async def wrapper(*inner_args, **kwargs):
        tool_call = await request.get_json()
        start_time = time.time()
        print(f"[DeepDiveToolServer] request={tool_call}")
        try:
            return await func(tool_call=tool_call, *inner_args, **kwargs)
        finally:
            print(f"[DeepDiveToolServer] elapsed={time.time() - start_time:.2f}s request={tool_call}")

    return wrapper


def _remote_info(tool_call):
    return tool_call.get("remote_env_info", {}) or {}


def _normalize_name(name: str) -> str:
    return name


def _forbidden_texts(remote_env_info):
    return remote_env_info.get("forbidden_texts", [])


@app.route("/health", methods=["GET"])
async def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["POST"])
@app.route("/tool", methods=["POST"])
@log_tool_call
async def call_tool(tool_call):
    session_id = str(tool_call.get("session_id") or "default")
    func_name = _normalize_name(tool_call.get("name", ""))
    arguments = tool_call.get("arguments", {}) or {}
    remote_env_info = _remote_info(tool_call)
    state = session2state[session_id]

    if func_name == "start_session":
        result = "Successfully started session."
    elif func_name == "close_session":
        session2state.pop(session_id, None)
        result = f"Successfully closed session {session_id}."
    elif func_name == "search":
        query = arguments.get("query", "")
        num = int(arguments.get("num", 10))
        if not query:
            result = "No query provided for search."
        else:
            result, idx2url = await search(
                query=query,
                num=num,
                forbidden_strs=_forbidden_texts(remote_env_info),
                proxy=args.http_proxy,
                serp_api_key=args.serp_api_key,
                serper_api_key=args.serper_api_key,
                provider=args.search_provider,
            )
            state["idx2url"] = idx2url
    elif func_name == "click":
        idx2url = state.get("idx2url", {})
        link_id = arguments.get("link_id")
        url = idx2url.get(link_id)
        if not url:
            result = "Must provide a valid link_id from the latest search results."
        else:
            result = await parse_url(
                url=url,
                forbidden_strs=_forbidden_texts(remote_env_info),
                proxy=args.http_proxy,
                jina_api_key=args.jina_api_key,
            )
        result = result[:10000]
    elif func_name == "open":
        url = arguments.get("url")
        if not url:
            result = "Must provide a valid url for opening."
        else:
            result = await parse_url(
                url=url,
                forbidden_strs=_forbidden_texts(remote_env_info),
                proxy=args.http_proxy,
                jina_api_key=args.jina_api_key,
            )
        result = result[:10000]
    else:
        result = f"Undefined function: {func_name}"

    return jsonify({"output": result, "observation": result})


@app.route("/search", methods=["POST"])
@app.route("/click", methods=["POST"])
@app.route("/open", methods=["POST"])
async def compat_tool():
    payload = await request.get_json()
    name = request.path.strip("/")
    payload = {
        "session_id": payload.get("session_id", "default"),
        "name": name,
        "arguments": payload.get("arguments", payload),
        "remote_env_info": payload.get("remote_env_info", {}),
    }
    with app.test_request_context("/tool", method="POST", json=payload):
        return await call_tool()


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

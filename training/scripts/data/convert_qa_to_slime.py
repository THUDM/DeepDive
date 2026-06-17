import argparse
import json
from pathlib import Path


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "num": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Open a search result by id.",
            "parameters": {
                "type": "object",
                "properties": {"link_id": {"type": "integer"}},
                "required": ["link_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Open an explicit URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are DeepDive, a deep-search agent. Use the DeepDive web tools to search, open pages, and inspect evidence. When you have enough evidence, answer the user directly and include sources when possible."""


def convert_row(row, source: str):
    question = row.get("question") or row.get("input") or row.get("prompt")
    answer = row.get("answer") or row.get("label") or row.get("target")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if not question or answer is None:
        raise ValueError(f"row must contain question and answer: {row}")
    return {
        "id": row.get("id", ""),
        "source": source,
        "input_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "label": str(answer),
        "tools": TOOLS,
        "metadata": {
            "remote_env_info": {
                "forbidden_texts": [question],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="deepdive")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(convert_row(json.loads(line), args.source), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

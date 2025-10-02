import os
import re
import json
import time
import openai
import argparse
import threading
from tqdm import tqdm
from dotenv import load_dotenv
from functools import partial
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import GENERATE_QUESTION_KILT

load_dotenv()


def build_client(api_key: Optional[str], base_url: str) -> Optional[openai.OpenAI]:
    if not api_key:
        print("[WARN] OPENAI_API_KEY is not set; run will skip LLM calls.")
        return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def extract_qa(text: str) -> Dict[str, Optional[str]]:
    if not isinstance(text, str):
        return {"question": None, "answer": None}
    qm = re.search(r"<question>:\s*(.*?)(?:\n|$)<answer>:",
                   text, re.DOTALL | re.IGNORECASE)
    am = re.search(r"<answer>:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    q = qm.group(1).strip() if qm else None
    a = am.group(1).strip() if am else None
    return {"question": q, "answer": a}


def render_prompt(relations: Any, nodes: Any) -> str:
    return GENERATE_QUESTION_KILT % {
        "path": str(relations),
        "intro": str(nodes),
    }


def call_llm(client: Optional[openai.OpenAI], model: str, prompt: str, retries: int, backoff_sec: float) -> Optional[str]:
    if not client:
        return None
    err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = e
            sleep_for = backoff_sec * attempt
            print(
                f"[LLM] error (attempt {attempt}/{retries}): {e}. sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    print(f"[LLM] failed after {retries} attempts: {err}")
    return None


def parse_jsonl_ids(path: str, id_key: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if id_key in obj and isinstance(obj[id_key], (str, int)):
                    done.add(str(obj[id_key]))
            except Exception:
                continue
    return done


def load_input_lines(path: str, id_key: str, processed_ids: Set[str]) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if id_key in obj:
                if str(obj[id_key]) in processed_ids:
                    continue
            lines.append(s)
    return lines


def process_line(
    line: str,
    client: Optional[openai.OpenAI],
    model: str,
    id_key: str,
    skip_disambiguation: bool,
    retries: int,
    backoff_sec: float,
) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(line)
    except Exception as e:
        print(f"[PARSE] invalid json: {e}")
        return None

    relations = data.get("relations", [])
    nodes = data.get("nodes", {})
    if skip_disambiguation and isinstance(relations, list) and any(
        isinstance(x, str) and "disambiguation" in x.lower() for x in relations
    ):
        return None

    prompt = render_prompt(relations, nodes)
    content = call_llm(client, model, prompt,
                       retries=retries, backoff_sec=backoff_sec)
    if not content:
        return None

    qa = extract_qa(content)
    out: Dict[str, Any] = {"question": qa["question"], "answer": qa["answer"]}
    if id_key in data:
        out[id_key] = data[id_key]
    return out


def merge_jsonl(files: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in files:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_kilt_questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=str, help="Input JSONL file")
    p.add_argument("--output", required=True, type=str,
                   help="Output JSONL file (append mode with checkpointing)")
    p.add_argument("--openrouter-api-key",
                   default=os.getenv("OPENROUTER_API_KEY"), type=str)
    p.add_argument("--openrouter-base-url", default=os.getenv("OPENROUTER_BASE_URL",
                   "https://openrouter.ai/api/v1"), type=str)
    p.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", "gemini-2.5-pro"), type=str
    )
    p.add_argument("--max-workers", default=128, type=int)
    p.add_argument("--future-timeout-sec", default=180, type=float)
    p.add_argument("--retries", default=3, type=int)
    p.add_argument("--backoff-sec", default=2.0, type=float)
    p.add_argument("--id-key", default="id", type=str)
    p.add_argument("--skip-disambiguation", action="store_true", default=True)
    p.add_argument("--no-skip-disambiguation",
                   dest="skip_disambiguation", action="store_false")
    p.add_argument("--merge-output", default="", type=str,
                   help="If set, merge multiple partial outputs into this file")
    p.add_argument("--merge-inputs", default="", type=str,
                   help="JSON array of JSONL file paths to merge")
    return p


def main():
    args = build_argparser().parse_args()

    client = build_client(args.openrouter_api_key, args.openrouter_base_url)

    processed_ids = parse_jsonl_ids(args.output, args.id_key)
    print(f"[RESUME] already processed ids: {len(processed_ids)}")

    lines = load_input_lines(args.input, args.id_key, processed_ids)
    total = len(lines)
    if total == 0:
        print("[DONE] nothing to process")
        if args.merge_output and args.merge_inputs:
            try:
                files = json.loads(args.merge_inputs)
                if isinstance(files, list):
                    merge_jsonl(files, args.merge_output)
                    print(f"[MERGED] -> {args.merge_output}")
            except Exception as e:
                print(f"[MERGE] invalid --merge-inputs: {e}")
        return

    write_lock = threading.Lock()
    fn = partial(
        process_line,
        client=client,
        model=args.model,
        id_key=args.id_key,
        skip_disambiguation=args.skip_disambiguation,
        retries=args.retries,
        backoff_sec=args.backoff_sec,
    )

    with open(args.output, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.max_workers) as ex, \
            tqdm(total=total, desc="Processing", ncols=80) as pbar:

        futures = [ex.submit(fn, line) for line in lines]
        for fut in as_completed(futures):
            res = None
            try:
                res = fut.result(timeout=args.future_timeout_sec)
            except Exception as e:
                print(f"[FUTURE] error/timeout: {e}")
            if res:
                with write_lock:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
            pbar.update(1)

    if args.merge_output and args.merge_inputs:
        try:
            files = json.loads(args.merge_inputs)
            if isinstance(files, list):
                merge_jsonl(files, args.merge_output)
                print(f"[MERGED] -> {args.merge_output}")
        except Exception as e:
            print(f"[MERGE] invalid --merge-inputs: {e}")


if __name__ == "__main__":
    main()

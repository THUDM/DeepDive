import os
import re
import json
import time
import openai
import random
import argparse
from tqdm import tqdm
import concurrent.futures
from dotenv import load_dotenv
from typing import Any, Dict, List, Set, Tuple
from kilt_query import KiltClient

load_dotenv()


def safe_search(pattern: str, text: Any, flags=0):
    if not isinstance(text, str):
        return None
    try:
        return re.search(pattern, text, flags=flags)
    except re.error:
        return None


def cut_see_also_and_filter_anchors(text: Any, anchors: Any):
    text = text if isinstance(text, str) else ""
    anchors = anchors if isinstance(anchors, list) else []
    patterns = [
        r'\bSee also\b\.?', r'\bReferences\b', r'\bExternal links\b',
        r'\bFurther reading\b', r'\bNotes\b', r'\bBibliography\b',
        r'\bSee also\s*:', r'参考资料', r'外部链接'
    ]
    match = None
    for pat in patterns:
        match = safe_search(pat, text, flags=re.IGNORECASE)
        if match:
            break
    if match:
        cut_pos = match.start()
        cut_text = text[:cut_pos]
        deleted_text = text[cut_pos:]
    else:
        cut_text = text
        deleted_text = ""
    keep_anchors = []
    for a in anchors:
        if not isinstance(a, dict):
            continue
        title = a.get("title") or a.get("href", "")
        title = title.replace('%20', ' ').replace("_", " ").replace("%23", "#")
        if title and title not in deleted_text:
            keep_anchors.append(a)
    return cut_text, keep_anchors


class RandomWalk:
    def __init__(self, kilt_client, llm_client, llm_model: str, mad_walk_prob: float = 0.5, mad_walk: int = 20):
        self.kilt_client = kilt_client
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.mad_walk_prob = mad_walk_prob
        self.mad_walk = mad_walk

    def query_llm(self, prompt: str) -> str:
        if not self.llm_client:
            return '0'
        for attempt in range(3):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a concise selector. Reply with only a number."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return resp.choices[0].message.content or '0'
            except Exception as e:
                print(f"[LLM] fail (Attempt {attempt+1}): {e}")
                time.sleep(1.5 * (attempt + 1))
        return '0'

    def _normalize_title(self, s: str) -> str:
        return (s or "").replace("%20", " ").replace("_", " ").replace("%23", "#").strip()

    def find_next_candidate(self, anchors, visited, title, current_text):
        anchors = anchors if isinstance(anchors, list) else []
        random.shuffle(anchors)
        suitable_candidates: List[Dict[str, Any]] = []
        candidate_descriptions: List[str] = []
        for a in anchors:
            if len(suitable_candidates) >= 5:
                break
            if not isinstance(a, dict):
                continue
            next_title = self._normalize_title(a.get("href", ""))
            if not next_title:
                continue
            next_title_lower = next_title.lower()
            anchor_text_lower = str(a.get("text", "")).lower()
            if next_title_lower in visited or anchor_text_lower in visited:
                continue
            try:
                candidate_anchors = self.kilt_client.query_relations(
                    next_title) or []
                candidate_text = self.kilt_client.query_text(next_title) or ""
            except Exception as e:
                print(f"[Next] fail: {e}")
                continue
            candidate_text, candidate_anchors = cut_see_also_and_filter_anchors(
                candidate_text, candidate_anchors)
            try:
                srch = self.kilt_client.kilt_search(next_title)
                next_wiki_title = srch["hits"]["hits"][0]["_source"].get(
                    "wikipedia_title", next_title) if srch and srch.get("hits", {}).get("hits") else next_title
            except Exception:
                next_wiki_title = next_title
            if (candidate_anchors and len(candidate_anchors) > 3 and
                next_wiki_title.lower() != (title or "").lower() and
                    next_title != title and '(disambiguation)' not in next_wiki_title.lower()):
                suitable_candidates.append(a)
                first_para = ""
                if isinstance(candidate_text, str) and candidate_text:
                    first_para = candidate_text.split('\n\n')[0]
                candidate_descriptions.append(
                    first_para if first_para else "No description available")
        if not suitable_candidates:
            return None
        if len(suitable_candidates) == 1:
            return suitable_candidates[0]
        prompt = [
            f"Current topic: {title}",
            f"Current context: {(current_text or '')[:600]}",
            "",
            "Please select the most logically connected next topic from the following candidates.",
            f"Respond with ONLY a number from 1 to {len(suitable_candidates)}.",
            ""
        ]
        for i, (cand, desc) in enumerate(zip(suitable_candidates, candidate_descriptions), 1):
            prompt.append(
                f"{i}. {cand.get('text', '[no anchor text]')}: {desc[:200]}..."
            )
        llm_choice = self.query_llm("\n".join(prompt))
        try:
            nums = re.findall(r'\b[1-5]\b', str(llm_choice))
            if nums:
                choice = int(nums[0])
                if 1 <= choice <= len(suitable_candidates):
                    return suitable_candidates[choice - 1]
            return suitable_candidates[0]
        except Exception:
            return suitable_candidates[0]

    def custom_random_walk(self,
                           start_node: str,
                           thread_id: int,
                           num_steps: int = 100,
                           max_visited: int = 100,
                           save_path: str = None,
                           save_every_k: int = 5,
                           mad_walk_range: Tuple[int, int] = (1, 3),
                           mad_walk_max_steps: int = 20) -> List[Dict[str, Any]]:
        all_subpaths: List[Dict[str, Any]] = []
        visited: Set[str] = set()
        history_nodes: Set[str] = set()
        failed_nodes: Set[str] = set()
        total_steps = 0
        current_node = start_node
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            f_jsonl = open(save_path, "a+", encoding="utf-8")
        else:
            f_jsonl = None
        pbar = tqdm(total=num_steps, desc=f"Thread-{thread_id} Random Walk")
        while total_steps < num_steps and len(visited) < max_visited:
            path: List[Dict[str, Any]] = []
            steps = 0
            while steps < save_every_k and total_steps < num_steps and len(visited) < max_visited:
                try:
                    text = self.kilt_client.query_text(current_node)
                    anchors = self.kilt_client.query_relations(current_node)
                    search_result = self.kilt_client.kilt_search(current_node)
                    if not (search_result and search_result.get("hits", {}).get("hits")):
                        raise ValueError(
                            f"Invalid search result for node: {current_node}")
                    title = search_result["hits"]["hits"][0]["_source"].get(
                        "wikipedia_title", current_node) or current_node
                    text, anchors = cut_see_also_and_filter_anchors(
                        text, anchors)
                except Exception as e:
                    print(f"[Node] visit failed {current_node}: {e}")
                    failed_nodes.add(current_node)
                    if len(history_nodes) > 1:
                        possible_starts = list(
                            history_nodes - {current_node} - failed_nodes)
                        if possible_starts:
                            current_node = random.choice(possible_starts)
                            path = []
                            break
                        else:
                            print("[Node] No valid history nodes available")
                            break
                    else:
                        print("[Node] No history nodes available")
                        break
                path.append({"step": steps, "title": title,
                            "text": text, "anchors": anchors})
                visited.add(title.lower())
                history_nodes.add(title)
                steps += 1
                total_steps += 1
                pbar.update(1)
                if not anchors:
                    break
                random.shuffle(anchors)
                next_candidate = self.find_next_candidate(
                    anchors=anchors, visited=visited, title=title, current_text=text)
                if not next_candidate:
                    print(
                        "[Path] invalidates the current sub-path and randomly selects a new starting point from history ")
                    path = []
                    if len(history_nodes) > 1:
                        possible_starts = list(history_nodes - {title})
                        current_node = random.choice(possible_starts)
                    else:
                        break
                    break
                else:
                    current_node = self._normalize_title(
                        next_candidate.get("href", current_node))
            if path and steps == save_every_k:
                relations = [node["title"] for node in path]
                unique_relations = list(dict.fromkeys(relations))
                if len(unique_relations) >= 7:
                    nodes = {node["title"]: node["text"] for node in path}
                    subpath_json = {
                        "relations": unique_relations, "nodes": nodes}
                    all_subpaths.append(subpath_json)
                    if f_jsonl:
                        f_jsonl.write(json.dumps(
                            subpath_json, ensure_ascii=False) + "\n")
                        f_jsonl.flush()
            if total_steps < num_steps and len(visited) < max_visited:
                mad_steps = random.randint(*mad_walk_range)
                found_qualified = False
                mad_walk_attempts = 0
                mad_walk_limit = mad_walk_max_steps
                temp_node = current_node
                while not found_qualified and mad_walk_attempts < mad_walk_limit:
                    curr = temp_node
                    for _ in range(mad_steps):
                        a_list = self.kilt_client.query_relations(curr)
                        if not a_list:
                            break
                        unvisited_anchors = [a for a in a_list if self._normalize_title(
                            a.get("href", "")).lower() not in visited]
                        if not unvisited_anchors:
                            break
                        next_anchor = random.choice(unvisited_anchors)
                        curr = self._normalize_title(
                            next_anchor.get("href", curr))
                    if curr.lower() in visited:
                        mad_walk_attempts += 1
                        continue
                    candidate_anchors = self.kilt_client.query_relations(curr) or [
                    ]
                    candidate_text = self.kilt_client.query_text(curr) or ""
                    candidate_text, candidate_anchors = cut_see_also_and_filter_anchors(
                        candidate_text, candidate_anchors)
                    if candidate_anchors and len(candidate_anchors) > 3:
                        found_qualified = True
                        current_node = curr
                        break
                    else:
                        mad_steps += 1
                        mad_walk_attempts += 1
                        temp_node = curr
                if not found_qualified:
                    unvisited_history = [
                        node for node in history_nodes if node.lower() not in visited]
                    if unvisited_history:
                        current_node = random.choice(unvisited_history)
                    else:
                        possible_starts = list(history_nodes)
                        if possible_starts:
                            current_node = random.choice(possible_starts)
                        else:
                            break
        pbar.close()
        if f_jsonl:
            f_jsonl.close()
        return all_subpaths


def merge_jsonl_files(file_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for fname in file_list:
            if not os.path.exists(fname):
                continue
            with open(fname, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


def parse_args():
    p = argparse.ArgumentParser(
        prog="random_walk_kilt", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--openai-api-key",
                   default=os.getenv("OPENAI_API_KEY"), type=str)
    p.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL",
                   "https://api.openai.com/v1"), type=str)
    p.add_argument("--openai-model",
                   default=os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14"), type=str)
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI",
                   "mongodb://localhost:27017"), type=str)
    p.add_argument(
        "--mongo-db", default=os.getenv("MONGO_DB", "kilt"), type=str)
    p.add_argument(
        "--mongo-coll", default=os.getenv("MONGO_COLL", "knowledgesource"), type=str)
    p.add_argument("--num-steps", default=4096, type=int)
    p.add_argument("--max-visited", default=4096, type=int)
    p.add_argument("--save-every-k", default=8, type=int)
    p.add_argument("--mad-walk-min", default=2, type=int)
    p.add_argument("--mad-walk-max", default=4, type=int)
    p.add_argument("--mad-walk-max-steps", default=20, type=int)
    p.add_argument("--max-workers", default=16, type=int)
    p.add_argument("--out-dir", default="./random_walk_outputs", type=str)
    p.add_argument("--merged-file", default="random_walk_all.jsonl", type=str)
    p.add_argument("--tasks-file", default="", type=str)
    p.add_argument("--tasks-json", default="", type=str)
    return p.parse_args()


def load_tasks(args) -> List[Dict[str, str]]:
    if args.tasks_json:
        try:
            tasks = json.loads(args.tasks_json)
            if isinstance(tasks, list):
                return tasks
        except Exception as e:
            print(f"[WARN] parse --tasks-json failed: {e}")
    if args.tasks_file and os.path.exists(args.tasks_file):
        try:
            with open(args.tasks_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            if isinstance(tasks, list):
                return tasks
        except Exception as e:
            print(f"[WARN] read --tasks-file failed: {e}")
    return [
        {"name": "movie", "start": "Quentin Tarantino",
            "filename": "random_walk_movie.jsonl"},
        {"name": "music", "start": "Beyonce",
            "filename": "random_walk_music.jsonl"},
        {"name": "architecture", "start": "Zaha Hadid",
            "filename": "random_walk_architecture.jsonl"},
        {"name": "media", "start": "Oprah Winfrey",
            "filename": "random_walk_media.jsonl"},
    ]


def build_llm_client(args):
    api_key = args.openai_api_key
    if not api_key:
        print(
            "[WARN] OPENAI_API_KEY not set. LLM selection will fallback to first candidate.")
        return None
    return openai.OpenAI(api_key=api_key, base_url=args.openai_base_url)


def worker(start_node, thread_id, save_path, args, llm_client):
    kilt_client = KiltClient(
        mongo_uri=args.mongo_uri,
        db_name=args.mongo_db,
        coll_name=args.mongo_coll,
        use_text_index=True,
        text_language=None,
    )
    rw = RandomWalk(kilt_client, llm_client=llm_client,
                    llm_model=args.openai_model)
    _ = rw.custom_random_walk(
        start_node=start_node,
        thread_id=thread_id,
        num_steps=args.num_steps,
        max_visited=args.max_visited,
        save_path=save_path,
        save_every_k=args.save_every_k,
        mad_walk_range=(args.mad_walk_min, args.mad_walk_max),
        mad_walk_max_steps=args.mad_walk_max_steps
    )
    return save_path


def main():
    args = parse_args()
    llm_client = build_llm_client(args)
    tasks = load_tasks(args)
    os.makedirs(args.out_dir, exist_ok=True)
    for t in tasks:
        t["filename"] = os.path.join(args.out_dir, t["filename"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for idx, task in enumerate(tasks):
            futures.append(executor.submit(
                worker, task["start"], idx, task["filename"], args, llm_client))
        for future in concurrent.futures.as_completed(futures):
            print(f"Finished: {future.result()}")
    print("All random walks finished. Merging all files...")
    all_files = [t["filename"] for t in tasks]
    merged_path = os.path.join(args.out_dir, args.merged_file) if not os.path.isabs(
        args.merged_file) else args.merged_file
    merge_jsonl_files(all_files, merged_path)
    print(f"Merged to: {merged_path}")


if __name__ == "__main__":
    main()

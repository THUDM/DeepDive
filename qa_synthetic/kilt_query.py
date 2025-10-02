import re
import time
import json
from typing import Optional, Dict, List, Any, Tuple
from urllib.parse import unquote

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError


class KiltClient:
    """
    Direct access to the KILT MongoDB knowledge base; aligned with your random-walk script.

    Public API:
      - query_text(title): returns a string (concatenate paragraphs until ≥min_chars)
      - query_relations(title): returns a list of anchors [{"href", "text", "title", ...}]
      - kilt_search(query): returns an Elasticsearch-like result shape
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "kilt",
        coll_name: str = "knowledgesource",
        max_retries: int = 3,
        retry_interval: float = 1.0,
        use_text_index: bool = True,
        text_language: Optional[str] = None,
        ensure_basic_indexes: bool = True,
    ) -> None:
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.coll_name = coll_name
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.use_text_index = use_text_index
        self.text_language = text_language  # fixed: removed stray char

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.col = self.db[self.coll_name]

        # Create a couple of lightweight indexes for common lookups.
        if ensure_basic_indexes:
            try:
                self.col.create_index([("wikipedia_id", ASCENDING)])
                self.col.create_index([("wikipedia_title", ASCENDING)])
            except Exception:
                # Index creation errors should not be fatal for read paths.
                pass

    def _retry(self, func, *args, **kwargs):
        """Run `func` with simple retry-on-PyMongoError logic."""
        last_err = None
        for _ in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except PyMongoError as e:
                last_err = e
                time.sleep(self.retry_interval)
        if last_err:
            raise last_err

    @staticmethod
    def _normalize_title(s: str) -> str:
        """URL-decode, trim, replace underscores with spaces, handle %23 → '#'."""
        return unquote(s or "").strip().replace("_", " ").replace("%23", "#")

    @staticmethod
    def _both_id_forms(doc_id: Any) -> Tuple[Optional[int], str]:
        """Return (int-form if possible, string-form) of a document id."""
        as_str = str(doc_id)
        try:
            as_int: Optional[int] = int(as_str)
        except Exception:
            as_int = None
        return as_int, as_str

    def _find_by_title(self, title_norm: str, projection=None) -> Optional[Dict[str, Any]]:
        """Find by exact, then case-insensitive exact, then fuzzy title match."""
        # 1) exact
        doc = self._retry(self.col.find_one, {
                          "wikipedia_title": title_norm}, projection)
        if doc:
            return doc
        # 2) case-insensitive whole-string
        doc = self._retry(
            self.col.find_one,
            {"wikipedia_title": {
                "$regex": f"^{re.escape(title_norm)}$", "$options": "i"}},
            projection,
        )
        if doc:
            return doc
        # 3) fuzzy substring
        doc = self._retry(
            self.col.find_one,
            {"wikipedia_title": {"$regex": re.escape(
                title_norm), "$options": "i"}},
            projection,
        )
        return doc

    def kilt_search(self, query: str, size: int = 10) -> Dict[str, Any]:
        """
        Return an ES-style result shape:
        {
          "hits": {
            "hits": [
              {"_source": {
                "wikipedia_id": ..., "wikipedia_title": ...,
                "first_paragraph": "...", "para_count": N, "anchor_count": M
              }}, ...
            ]
          }
        }
        """
        query = self._normalize_title(query)
        size = max(1, min(size, 50))
        results: List[Dict[str, Any]] = []

        projection_preview = {
            "_id": 0,
            "wikipedia_id": 1,
            "wikipedia_title": 1,
            "text": {"$slice": 1},
            "anchors": 1,
        }

        # 1) exact title
        exact = list(
            self._retry(self.col.find, {
                        "wikipedia_title": query}, projection_preview).limit(size)
        )
        results.extend(exact)

        # 2) $text search with textScore ordering
        if len(results) < size and self.use_text_index:
            try:
                remaining = size - len(results)
                text_filter: Dict[str, Any] = {"$text": {"$search": query}}
                if self.text_language:
                    text_filter["$text"]["$language"] = self.text_language
                text_proj = {**projection_preview,
                             "score": {"$meta": "textScore"}}
                cur = (
                    self._retry(self.col.find, text_filter, text_proj)
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(remaining)
                )
                results.extend(list(cur))
            except PyMongoError:
                # Fall through to regex fallback.
                pass

        # 3) regex fallback over title and first page of text
        if len(results) < size:
            remaining = size - len(results)
            q_esc = re.escape(query)
            rx_filter = {
                "$or": [
                    {"wikipedia_title": {"$regex": q_esc, "$options": "i"}},
                    {"text": {"$elemMatch": {"$regex": q_esc, "$options": "i"}}},
                ]
            }
            cur = self._retry(self.col.find, rx_filter,
                              projection_preview).limit(remaining)
            results.extend(list(cur))

        hits: List[Dict[str, Any]] = []
        for r in results[:size]:
            first_para: Optional[str] = None
            if isinstance(r.get("text"), list) and r["text"]:
                first_para = r["text"][0]
            elif isinstance(r.get("text"), str):
                first_para = r["text"]

            anchors = r.get("anchors", [])
            anchor_count = len(anchors) if isinstance(
                anchors, list) else (anchors or 0)
            para_count = (
                len(r["text"]) if isinstance(r.get("text"),
                                             list) else (1 if first_para else 0)
            )

            hits.append(
                {
                    "_source": {
                        "wikipedia_id": r.get("wikipedia_id"),
                        "wikipedia_title": r.get("wikipedia_title"),
                        "first_paragraph": first_para,
                        "para_count": para_count,
                        "anchor_count": anchor_count,
                    }
                }
            )

        return {"hits": {"hits": hits}}

    def kilt_open(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Open a full document by wikipedia_id (int or str) or by title."""
        as_int, as_str = self._both_id_forms(doc_id)
        or_list: List[Dict[str, Any]] = []
        if as_int is not None:
            or_list.append({"wikipedia_id": as_int})
        or_list.append({"wikipedia_id": as_str})

        doc = self._retry(self.col.find_one, {"$or": or_list}, {"_id": 0})
        if doc:
            return doc

        # Treat as title lookup
        title_norm = self._normalize_title(doc_id)
        return self._find_by_title(title_norm, {"_id": 0})

    def query_text(
        self,
        title: str,
        min_chars: int = 500,
        max_paragraphs: int = 6,
        hard_max_chars: int = 5000,
    ) -> str:
        """
        Return a sufficiently long text snippet by concatenating paragraphs:
          - Append paragraphs from `text` (list[str]) until ≥ min_chars or `max_paragraphs`.
          - Always return a string. If not found, return "" (avoids TypeError in regex ops).
        """
        title_norm = self._normalize_title(title)
        doc = self._find_by_title(title_norm, {"_id": 0, "text": 1})
        if not doc:
            return ""

        text_field = doc.get("text")
        if isinstance(text_field, str):
            s = text_field.strip()
            return s[:hard_max_chars] if s else ""

        if isinstance(text_field, list) and text_field:
            buff: List[str] = []
            total = 0
            for i, para in enumerate(text_field):
                if not isinstance(para, str) or not para.strip():
                    continue
                buff.append(para.strip())
                total += len(para)
                if total >= min_chars or i + 1 >= max_paragraphs:
                    break
            joined = "\n\n".join(buff).strip()
            return joined[:hard_max_chars] if joined else ""

        return ""

    def query_relations(self, title: str) -> List[Dict[str, Any]]:
        """
        Return a normalized list of anchors (each contains at least href and text).
        - `href` / `title` are normalized to Wikipedia-title style (decoded, underscores → spaces).
        - Always return a list (empty list when not found).
        """
        title_norm = self._normalize_title(title)
        doc = self._find_by_title(title_norm, {"_id": 0, "anchors": 1})
        if not doc:
            return []

        anchors = doc.get("anchors")
        if not isinstance(anchors, list):
            return []

        out: List[Dict[str, Any]] = []
        for a in anchors:
            if not isinstance(a, dict):
                continue
            raw_href = a.get("href") or a.get("title") or ""
            href_title = self._normalize_title(raw_href)
            text = a.get("text") or a.get("anchor_text") or ""
            title_field = a.get("title") or href_title
            out.append(
                {
                    "href": href_title,         # key used by your code
                    "text": str(text),          # key used by your code
                    "title": self._normalize_title(title_field),
                    # Preserve any other original fields for potential downstream use.
                    **{k: v for k, v in a.items() if k not in ("href", "text", "title")},
                }
            )
        return out


def run_smoke_tests() -> None:
    client = KiltClient(
        mongo_uri="mongodb://localhost:27017",
        db_name="kilt",
        coll_name="knowledgesource",
        use_text_index=True,          # leverage your text index
        text_language="english",      # if index uses default_language: "none", set to "none"
        ensure_basic_indexes=True,
    )

    tests = [
        # 1) direct title (should match exactly)
        "TCU Horned Frogs football",
        # 2) URL-encoded title → should match after normalization
        "Jyv%C3%A4skyl%C3%A4",
        # 3) list-style title (may rely on text index / regex)
        "List of architecture prizes",
        # 4) generic keywords (tests $text)
        "architect association Finland",
    ]

    for q in tests:
        print("\n==============================")
        print("QUERY:", q)
        # Search (prefers exact → text → regex)
        res = client.kilt_search(q, size=5)
        print("[kilt_search top5]")
        print(json.dumps(res, ensure_ascii=False, indent=2))

        # Text snippet
        txt = client.query_text(q)
        print("[query_text]")
        print(txt if txt is not None else "None")

        # Anchors
        rel = client.query_relations(q)
        print("[query_relations]")
        if rel is None:
            print("None")
        else:
            preview = rel[:3] if isinstance(rel, list) else rel
            print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_smoke_tests()

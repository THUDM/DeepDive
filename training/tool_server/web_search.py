import asyncio
import re
from typing import Dict, Iterable, List, Tuple

import aiohttp


def normalize_string(text: str) -> str:
    pattern = r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]"
    return " ".join(re.findall(pattern, text.lower().strip()))


def _word_ngrams(text: str, n: int) -> set:
    words = text.split()
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def contains_forbidden_text(text: str, forbidden: str, ngram_size: int = 13) -> bool:
    if not text.strip() or not forbidden.strip():
        return False
    normalized_text = normalize_string(text)
    normalized_forbidden = normalize_string(forbidden)
    if len(normalized_forbidden.split()) < ngram_size:
        return normalized_forbidden in normalized_text
    forbidden_ngrams = _word_ngrams(normalized_forbidden, ngram_size)
    return bool(forbidden_ngrams & _word_ngrams(normalized_text, ngram_size))


def _allowed(block: str, forbidden_texts: Iterable[str]) -> bool:
    return not any(contains_forbidden_text(block, text) for text in forbidden_texts)


async def search(
    query: str,
    num: int = 10,
    forbidden_strs: List[str] = None,
    proxy: str = None,
    retry_times: int = 3,
    serp_api_key: str = None,
    serper_api_key: str = None,
    provider: str = "serpapi",
) -> Tuple[str, Dict[int, str]]:
    forbidden_strs = forbidden_strs or []
    timeout = aiohttp.ClientTimeout(total=30)
    last_error = None

    for attempt in range(retry_times):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if provider == "serper":
                    if not serper_api_key:
                        raise ValueError("SERPER_API_KEY is required when SEARCH_PROVIDER=serper")
                    async with session.post(
                        "https://google.serper.dev/search",
                        json={"q": query, "num": num},
                        headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
                        proxy=proxy,
                        ssl=False,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    results = data.get("organic", [])
                else:
                    if not serp_api_key:
                        raise ValueError("SERP_API_KEY is required when SEARCH_PROVIDER=serpapi")
                    params = {
                        "q": query,
                        "num": num,
                        "engine": "google",
                        "api_key": serp_api_key,
                    }
                    async with session.get(
                        "https://serpapi.com/search.json",
                        params=params,
                        proxy=proxy,
                        ssl=False,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    results = data.get("organic_results", [])
            blocks = []
            idx2url = {}
            for item in results:
                title = item.get("title", "")
                url = item.get("link", "")
                snippet = item.get("snippet", "")
                raw_block = f"{title}\n{url}\n{snippet}"
                if not url or not _allowed(raw_block, forbidden_strs):
                    continue
                idx = len(blocks) + 1
                idx2url[idx] = url
                blocks.append(
                    f"【{idx}】{title}\n"
                    f"{url}\n"
                    f"{snippet}\n"
                )
            return "\n".join(blocks).strip() or "No results found.", idx2url
        except (asyncio.TimeoutError, Exception) as exc:
            last_error = exc
            if attempt + 1 < retry_times:
                await asyncio.sleep(1)

    return f"Failed to fetch search results. Last error: {last_error}", {}


async def parse_url(
    url: str,
    forbidden_strs: List[str] = None,
    proxy: str = None,
    retry_times: int = 3,
    jina_api_key: str = None,
) -> str:
    if not jina_api_key:
        raise ValueError("JINA_API_KEY is required")
    forbidden_strs = forbidden_strs or []
    url = str(url).replace("view-source:", "")
    url = url.replace("https://r.jina.ai/", "").replace("http://r.jina.ai/", "")
    headers = {"Authorization": f"Bearer {jina_api_key}"}
    timeout = aiohttp.ClientTimeout(total=30)
    last_error = None

    for attempt in range(retry_times):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    proxy=proxy,
                    ssl=False,
                ) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
            if not _allowed(text, forbidden_strs):
                return "Failed: forbidden string found in page content."
            return text
        except (asyncio.TimeoutError, Exception) as exc:
            last_error = exc
            if attempt + 1 < retry_times:
                await asyncio.sleep(1)

    return f"Failed to parse URL content. Last error: {last_error}"

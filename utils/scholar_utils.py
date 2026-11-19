# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paper search and verification across several free scholarly databases.

Sources are tried in order (default: Semantic Scholar, OpenAlex, arXiv) and
the next one is used when a source is rate-limited, down, or has no match.
A source that keeps failing is skipped for a cooldown period. Every result is
normalised to the Semantic Scholar record shape used by the agents:

    {"title", "authors": [{"name"}], "venue", "year", "abstract",
     "citationCount", "journal": {"name", "volume", "pages"} | None,
     "publicationDate", "arxivId", "doi", "source"}

Configuration (environment / .env):
    PAPER_SEARCH_SOURCES        comma list, default "s2,openalex,arxiv"
    SEMANTIC_SCHOLAR_API_KEY    optional
    OPENALEX_API_KEY            optional (keyless: ~100 searches/day per IP)
"""

import datetime
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional

import requests
from thefuzz import fuzz  # type: ignore

from dotenv import load_dotenv  # type: ignore

cur_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(cur_dir, "../.env"))

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,venue,year,abstract,citationCount,journal,publicationDate,externalIds"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = (
    "title,publication_year,publication_date,cited_by_count,authorships,"
    "primary_location,abstract_inverted_index,biblio,ids,type"
)
ARXIV_API_URL = "https://export.arxiv.org/api/query"

DEFAULT_SOURCES = ("s2", "openalex", "arxiv")
# Seconds a source is skipped after it exhausted its retries.
SOURCE_COOLDOWN = float(os.environ.get("PAPER_SEARCH_COOLDOWN", "300"))
# Fuzzy title-match threshold (0-100) for verification.
TITLE_MATCH_THRESHOLD = 70


class SourceUnavailable(Exception):
    """A source is rate-limited, down or unreachable (as opposed to 'no results')."""

    def __init__(self, reason: str, retry_after: Optional[float] = None):
        super().__init__(reason)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Rate limiting, cooldown, caching
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Spaces requests to one source evenly across all threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            self._next_time = max(now, self._next_time) + self.min_interval
        if wait > 0:
            time.sleep(wait)


_limiters = {
    # S2: ~1 req/s with a key; the keyless pool is shared, so be gentle.
    "s2": _RateLimiter(float(os.environ.get("SEMANTIC_SCHOLAR_MIN_INTERVAL", "1.0"))),
    # OpenAlex search endpoints refuse bursts; space them out.
    "openalex": _RateLimiter(float(os.environ.get("OPENALEX_MIN_INTERVAL", "0.5"))),
    # arXiv API terms: no more than one request every 3 seconds.
    "arxiv": _RateLimiter(float(os.environ.get("ARXIV_MIN_INTERVAL", "3.0"))),
}

_cooldown_until: Dict[str, float] = {}
_cooldown_lock = threading.Lock()

# Records already fetched by keyword search, keyed by normalised title, so the
# later verification step needs no extra request.
_paper_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _cache_put(records: List[Dict]):
    with _cache_lock:
        for r in records:
            _paper_cache.setdefault(_normalize_title(r["title"]), r)


def _cache_get(title: str) -> Optional[Dict]:
    with _cache_lock:
        return _paper_cache.get(_normalize_title(title))


def _in_cooldown(source: str) -> bool:
    with _cooldown_lock:
        return _cooldown_until.get(source, 0.0) > time.monotonic()


def _start_cooldown(source: str, reason: str, seconds: Optional[float] = None):
    seconds = SOURCE_COOLDOWN if seconds is None else seconds
    with _cooldown_lock:
        already = _cooldown_until.get(source, 0.0) > time.monotonic()
        _cooldown_until[source] = max(
            _cooldown_until.get(source, 0.0), time.monotonic() + seconds
        )
    if not already:
        print(
            f"      ⚠️ Paper source '{source}' unavailable ({reason}); "
            f"using fallbacks for the next {int(seconds)}s."
        )


def _retry_after_seconds(rsp: requests.Response) -> Optional[float]:
    value = rsp.headers.get("Retry-After", "")
    if value.replace(".", "", 1).isdigit():
        return float(value)
    try:  # OpenAlex puts it in the JSON body
        body = rsp.json()
        value = body.get("retryAfter") if isinstance(body, dict) else None
        return float(value) if value is not None else None
    except ValueError:
        return None


def _http_get(
    source: str,
    url: str,
    params: Dict,
    headers: Optional[Dict] = None,
    max_retries: int = 3,
    timeout: int = 15,
) -> Optional[requests.Response]:
    """GET with per-source rate limiting and backoff.

    Returns the response on 200, None on a non-retryable client error (bad
    query), and raises SourceUnavailable on persistent 429/5xx/network errors.
    """
    last = "unknown error"
    retry_after = None
    for attempt in range(max_retries):
        _limiters[source].wait()
        try:
            rsp = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        except requests.RequestException as e:
            last = f"{type(e).__name__}"
        else:
            if rsp.status_code == 200:
                return rsp
            if rsp.status_code != 429 and rsp.status_code < 500:
                return None
            last = f"HTTP {rsp.status_code}"
            retry_after = _retry_after_seconds(rsp)
            if retry_after is not None and retry_after > 10:
                # Long server-requested wait (overload, daily budget): fall
                # back to the next source instead of blocking.
                break
            if retry_after is not None:
                time.sleep(retry_after)
                continue
        if attempt + 1 < max_retries:
            time.sleep(min(2 ** (attempt + 1), 20))
    raise SourceUnavailable(last, retry_after)


def is_date_valid(p_date, p_year, cutoff_str) -> bool:
    if not cutoff_str:
        return True
    try:
        c_year, c_month = map(int, cutoff_str.split("-")[:2])
        cutoff_dt = datetime.datetime(c_year, c_month, 1)
        if p_date:
            try:
                p_dt = datetime.datetime.strptime(p_date[:10], "%Y-%m-%d")
                return p_dt < cutoff_dt
            except ValueError:
                pass
        if p_year:
            if p_year < c_year:
                return True
            if p_year == c_year and c_month > 1:
                return True
            return False
        return True
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def _s2_normalize(r: Dict) -> Dict:
    ext = r.get("externalIds") or {}
    out = dict(r)
    out["arxivId"] = ext.get("ArXiv")
    out["doi"] = ext.get("DOI")
    out["source"] = "s2"
    return out


def _s2_search(query: str, limit: int) -> List[Dict]:
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}
    rsp = _http_get(
        "s2",
        S2_SEARCH_URL,
        {"query": query, "limit": limit, "fields": S2_FIELDS},
        headers=headers,
        max_retries=4,
    )
    if rsp is None:
        return []
    return [_s2_normalize(r) for r in (rsp.json().get("data") or []) if r.get("title")]


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def _openalex_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, idxs in inverted.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def _openalex_normalize(w: Dict) -> Optional[Dict]:
    title = w.get("title")
    if not title:
        return None
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    venue = src.get("display_name") or ""
    landing = loc.get("landing_page_url") or ""
    arxiv_match = re.search(r"arxiv\.org/abs/([^\s/?#]+?)(v\d+)?$", landing)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    if arxiv_id or venue.lower().startswith("arxiv"):
        venue = "arXiv"
    biblio = w.get("biblio") or {}
    pages = None
    if biblio.get("first_page"):
        pages = biblio["first_page"]
        if biblio.get("last_page"):
            pages += f"--{biblio['last_page']}"
    journal = None
    if src.get("type") == "journal" and venue != "arXiv":
        journal = {"name": venue, "volume": biblio.get("volume"), "pages": pages}
    doi = (w.get("ids") or {}).get("doi") or ""
    return {
        "title": title,
        "authors": [
            {"name": (a.get("author") or {}).get("display_name", "")}
            for a in (w.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ],
        "venue": venue,
        "year": w.get("publication_year"),
        "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
        "citationCount": w.get("cited_by_count") or 0,
        "journal": journal,
        "publicationDate": w.get("publication_date"),
        "arxivId": arxiv_id,
        "doi": doi.replace("https://doi.org/", "") or None,
        "source": "openalex",
    }


def _openalex_get(params: Dict) -> List[Dict]:
    params = dict(params, select=OPENALEX_SELECT)
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    rsp = _http_get("openalex", OPENALEX_WORKS_URL, params)
    if rsp is None:
        return []
    return [r for r in map(_openalex_normalize, rsp.json().get("results") or []) if r]


def _openalex_search(query: str, limit: int) -> List[Dict]:
    # Title+abstract only: the default `search` also matches full text, which
    # surfaces many loosely related papers.
    clean = re.sub(r"[,:|]", " ", query)
    return _openalex_get(
        {"filter": f"title_and_abstract.search:{clean}", "per_page": min(limit, 50)}
    )


def _openalex_title_search(title: str, limit: int) -> List[Dict]:
    # Commas/colons/pipes are filter syntax in OpenAlex.
    clean = re.sub(r"[,:|]", " ", title)
    return _openalex_get({"filter": f"title.search:{clean}", "per_page": limit})


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


# arXiv's search returns nothing when a stopword is one of the AND-ed terms.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "to", "via", "with", "without",
    "we", "our", "its", "this", "using", "towards", "toward",
}


def _arxiv_terms(text: str, field: str) -> str:
    words = [
        w
        for w in re.findall(r"[A-Za-z0-9]+", text)
        if len(w) > 1 and w.lower() not in _STOPWORDS
    ]
    return " AND ".join(f"{field}:{w}" for w in words[:12])


def _arxiv_query(search_query: str, limit: int) -> List[Dict]:
    if not search_query:
        return []
    rsp = _http_get(
        "arxiv",
        ARXIV_API_URL,
        {"search_query": search_query, "max_results": limit, "sortBy": "relevance"},
        timeout=30,
    )
    if rsp is None:
        return []
    root = ET.fromstring(rsp.content)
    out = []
    for e in root.findall(f"{_ATOM}entry"):
        title = " ".join((e.findtext(f"{_ATOM}title") or "").split())
        if not title:
            continue
        raw_id = (e.findtext(f"{_ATOM}id") or "").rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        published = (e.findtext(f"{_ATOM}published") or "")[:10]
        out.append(
            {
                "title": title,
                "authors": [
                    {"name": a.findtext(f"{_ATOM}name")}
                    for a in e.findall(f"{_ATOM}author")
                    if a.findtext(f"{_ATOM}name")
                ],
                "venue": "arXiv",
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "abstract": " ".join((e.findtext(f"{_ATOM}summary") or "").split()),
                "citationCount": 0,  # arXiv has no citation counts
                "journal": None,
                "publicationDate": published or None,
                "arxivId": arxiv_id or None,
                "doi": e.findtext(f"{_ARXIV}doi"),
                "source": "arxiv",
            }
        )
    return out


def _arxiv_search(query: str, limit: int) -> List[Dict]:
    return _arxiv_query(_arxiv_terms(query, "all"), limit)


def _arxiv_title_search(title: str, limit: int) -> List[Dict]:
    phrase = " ".join(re.findall(r"[A-Za-z0-9]+", title))
    results = _arxiv_query(f'ti:"{phrase}"', limit) if phrase else []
    return results or _arxiv_query(_arxiv_terms(title, "ti"), limit)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_KEYWORD_SEARCH: Dict[str, Callable[[str, int], List[Dict]]] = {
    "s2": _s2_search,
    "openalex": _openalex_search,
    "arxiv": _arxiv_search,
}
_TITLE_SEARCH: Dict[str, Callable[[str, int], List[Dict]]] = {
    "s2": lambda t, n: _s2_search(t, n),
    "openalex": _openalex_title_search,
    "arxiv": _arxiv_title_search,
}


def get_sources() -> List[str]:
    raw = os.environ.get("PAPER_SEARCH_SOURCES", ",".join(DEFAULT_SOURCES))
    sources = [s.strip().lower() for s in raw.split(",") if s.strip()]
    unknown = [s for s in sources if s not in _KEYWORD_SEARCH]
    if unknown:
        raise ValueError(
            f"Unknown paper source(s) {unknown}; choose from {list(_KEYWORD_SEARCH)}"
        )
    return sources


def _run_chain(registry, query: str, limit: int, accept: Callable[[List[Dict]], Optional[object]]):
    for source in get_sources():
        if _in_cooldown(source):
            continue
        try:
            results = registry[source](query, limit)
        except SourceUnavailable as e:
            _start_cooldown(source, str(e), e.retry_after)
            continue
        except Exception as e:  # parsing errors etc.: try next source
            print(f"      ⚠️ Paper source '{source}' error for '{query[:60]}': {e}")
            continue
        _cache_put(results)
        accepted = accept(results)
        if accepted:
            return accepted
    return None


def keyword_search(query: str, limit: int = 20, cutoff_date: str = None) -> List[Dict]:
    """Keyword search, falling back across sources until one returns results."""

    def accept(results):
        return [
            r
            for r in results
            if is_date_valid(r.get("publicationDate"), r.get("year"), cutoff_date)
        ]

    return _run_chain(_KEYWORD_SEARCH, query, limit, accept) or []


def _best_title_match(results: List[Dict], title: str, year_hint, cutoff_date) -> Optional[Dict]:
    best, best_ratio = None, 0
    for r in results:
        if not is_date_valid(r.get("publicationDate"), r.get("year"), cutoff_date):
            continue
        ratio = fuzz.ratio(title.lower(), r["title"].lower())
        if year_hint and r.get("year") == year_hint:
            ratio += 10
        if ratio > best_ratio:
            best, best_ratio = r, ratio
    return best if best_ratio > TITLE_MATCH_THRESHOLD else None


def title_search(title: str, year_hint: int = None, cutoff_date: str = None) -> Optional[Dict]:
    """Finds the record for a specific paper title, verifying it exists."""
    if not title:
        return None
    cached = _cache_get(title)
    if cached and is_date_valid(cached.get("publicationDate"), cached.get("year"), cutoff_date):
        return cached
    return _run_chain(
        _TITLE_SEARCH,
        title,
        5,
        lambda results: _best_title_match(results, title, year_hint, cutoff_date),
    )


# Backwards-compatible names.
s2_keyword_search = keyword_search
s2_title_search = title_search

print(
    "Paper sources: "
    + ", ".join(get_sources())
    + f" (Semantic Scholar key: {'set' if S2_API_KEY else 'not set'}, "
    + f"OpenAlex key: {'set' if OPENALEX_API_KEY else 'not set'})"
)

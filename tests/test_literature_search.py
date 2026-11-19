import json

import pytest

from utils import scholar_utils as su


class _Rsp:
    def __init__(self, status, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._payload


RESNET_S2 = {
    "title": "Deep Residual Learning for Image Recognition", "year": 2016,
    "publicationDate": "2016-06-27", "abstract": "ResNet.", "venue": "CVPR",
    "citationCount": 200000, "authors": [{"name": "Kaiming He"}],
    "journal": None, "externalIds": {"ArXiv": "1512.03385", "DOI": "10.1109/CVPR.2016.90"},
}
FUTURE_S2 = {
    "title": "A Future Paper", "year": 2030, "publicationDate": "2030-01-01",
    "abstract": "Too new.", "venue": "X", "citationCount": 0, "authors": [],
}

OPENALEX_WORK = {
    "title": "Attention Is All You Need",
    "publication_year": 2017,
    "publication_date": "2017-06-12",
    "cited_by_count": 150000,
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
    "primary_location": {
        "landing_page_url": "https://arxiv.org/abs/1706.03762v7",
        "source": {"display_name": "arXiv (Cornell University)", "type": "repository"},
    },
    "abstract_inverted_index": {"The": [0], "dominant": [1], "models": [2]},
    "biblio": {},
    "ids": {"doi": None},
}

ARXIV_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1512.03385v1</id>
    <published>2015-12-10T19:51:55Z</published>
    <title>Deep Residual Learning for
      Image Recognition</title>
    <summary>Deeper neural networks are more difficult to train.</summary>
    <author><name>Kaiming He</name></author>
    <author><name>Xiangyu Zhang</name></author>
  </entry>
</feed>"""


@pytest.fixture(autouse=True)
def fast_and_clean(monkeypatch):
    monkeypatch.setattr(su.time, "sleep", lambda s: None)
    for lim in su._limiters.values():
        monkeypatch.setattr(lim, "min_interval", 0.0)
    monkeypatch.delenv("PAPER_SEARCH_SOURCES", raising=False)
    su._paper_cache.clear()
    su._cooldown_until.clear()


def route(monkeypatch, handlers):
    """handlers: {url_substring: callable(params) -> _Rsp}; records calls."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        for key, fn in handlers.items():
            if key in url:
                calls.append((key, params))
                return fn(params)
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(su.requests, "get", fake_get)
    return calls


def test_s2_keyword_search_filters_cutoff_retries_429_and_caches(monkeypatch):
    responses = [_Rsp(429), _Rsp(200, {"data": [RESNET_S2, FUTURE_S2]})]
    calls = route(monkeypatch, {"semanticscholar": lambda p: responses.pop(0)})
    results = su.keyword_search("resnet", cutoff_date="2024-10")
    assert [r["title"] for r in results] == [RESNET_S2["title"]]
    assert results[0]["arxivId"] == "1512.03385" and results[0]["source"] == "s2"
    assert len(calls) == 2

    # verification is served from cache, no extra request
    hit = su.title_search(RESNET_S2["title"], 2016, "2024-10")
    assert hit["venue"] == "CVPR" and len(calls) == 2


def test_fallback_to_openalex_then_cooldown_skips_s2(monkeypatch):
    calls = route(monkeypatch, {
        "semanticscholar": lambda p: _Rsp(429),
        "openalex": lambda p: _Rsp(200, {"results": [OPENALEX_WORK]}),
    })
    hit = su.title_search("Attention Is All You Need", 2017, "2024-10")
    assert hit["source"] == "openalex"
    assert hit["venue"] == "arXiv" and hit["arxivId"] == "1706.03762"
    assert hit["abstract"] == "The dominant models"
    assert hit["authors"] == [{"name": "Ashish Vaswani"}]
    s2_calls = sum(1 for k, _ in calls if k == "semanticscholar")

    su._paper_cache.clear()
    su.keyword_search("transformers attention")
    # S2 is in cooldown, so it is not contacted again
    assert sum(1 for k, _ in calls if k == "semanticscholar") == s2_calls
    assert calls[-1][1]["filter"].startswith("title_and_abstract.search:")


def test_long_retry_after_falls_back_immediately(monkeypatch):
    calls = route(monkeypatch, {
        "semanticscholar": lambda p: _Rsp(404),
        "openalex": lambda p: _Rsp(429, {"retryAfter": 35}),
        "arxiv": lambda p: _Rsp(200, content=ARXIV_XML),
    })
    hit = su.title_search("Deep Residual Learning for Image Recognition", 2016, "2024-10")
    assert hit["source"] == "arxiv"
    assert sum(1 for k, _ in calls if k == "openalex") == 1  # no retries on 35s wait
    assert su._in_cooldown("openalex")


def test_arxiv_parsing_and_queries(monkeypatch):
    calls = route(monkeypatch, {"arxiv": lambda p: _Rsp(200, content=ARXIV_XML)})
    monkeypatch.setenv("PAPER_SEARCH_SOURCES", "arxiv")
    hit = su.title_search("Deep Residual Learning for Image Recognition", 2016, "2024-10")
    assert hit["title"] == "Deep Residual Learning for Image Recognition"
    assert hit["arxivId"] == "1512.03385" and hit["year"] == 2015
    assert hit["publicationDate"] == "2015-12-10"
    assert hit["authors"][1]["name"] == "Xiangyu Zhang"
    assert calls[0][1]["search_query"] == 'ti:"Deep Residual Learning for Image Recognition"'

    su._paper_cache.clear()
    su.keyword_search("learning for the image recognition")
    assert calls[-1][1]["search_query"] == "all:learning AND all:image AND all:recognition"


def test_no_results_anywhere(monkeypatch):
    route(monkeypatch, {
        "semanticscholar": lambda p: _Rsp(200, {"data": []}),
        "openalex": lambda p: _Rsp(200, {"results": []}),
        "arxiv": lambda p: _Rsp(200, content=b"<feed xmlns='http://www.w3.org/2005/Atom'/>"),
    })
    assert su.keyword_search("nothing") == []
    assert su.title_search("Nothing At All") is None


def test_unknown_source_rejected(monkeypatch):
    monkeypatch.setenv("PAPER_SEARCH_SOURCES", "s2,scopus")
    with pytest.raises(ValueError):
        su.get_sources()


def _agent(tmp_path, **kw):
    from methods.agents.literature_review_agent import HybridLiteratureAgent

    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text("x")
    return HybridLiteratureAgent(
        idea_path=str(tmp_path / "a"), experimental_log_path=str(tmp_path / "b"),
        latex_template_path=str(tmp_path / "c"), conference_guidelines_path=str(tmp_path / "d"),
        output_dir=str(tmp_path / "out"), **kw,
    )


def test_db_discovery_flow(monkeypatch, tmp_path):
    from methods.agents import literature_review_agent as lra

    paper = su._s2_normalize(RESNET_S2)
    monkeypatch.setattr(lra, "keyword_search", lambda q, limit, cutoff_date: [paper])
    llm_outputs = [
        {"queries": ["residual networks", "deep residual learning"]},
        {"section_name": "Intro", "candidates": [
            {"title": paper["title"], "year": 2016, "reason": "backbone"}]},
    ]
    prompts = []

    def fake_call(model_name, contents, generation_configs=None, **kwargs):
        prompts.append((model_name, contents[0]))
        out = llm_outputs.pop(0)
        return {"raw_response": json.dumps(out), "parsed_response": out}

    monkeypatch.setattr(lra, "call_llm_with_contents", fake_call)
    agent = _agent(tmp_path, model_name="ollama_chat/qwen3:8b", fast_model_name="ollama_chat/qwen3:8b")
    assert agent.search_backend == "s2"
    cands = agent._discover_candidates(
        {"section": "Intro", "focus": "residual nets", "context": "c", "search_type": "exploration"},
        {"intro_related_work_plan": {}},
        "2024-10",
    )
    assert cands[0]["title"] == paper["title"]
    assert all(m == "ollama_chat/qwen3:8b" for m, _ in prompts)
    assert paper["title"] in prompts[1][1]


def test_google_backend_falls_back_for_non_gemini(tmp_path):
    assert _agent(tmp_path, fast_model_name="openai/gpt-5", search_backend="google").search_backend == "s2"
    assert _agent(tmp_path, fast_model_name="gemini-3-flash-preview").search_backend == "google"


def test_preprint_bibtex(monkeypatch, tmp_path):
    from methods.agents import literature_review_agent as lra

    agent = _agent(tmp_path)
    registry = {}
    monkeypatch.setattr(lra, "title_search", lambda t, y, c: {
        "title": "Attention Is All You Need", "authors": [{"name": "Ashish Vaswani"}],
        "venue": "arXiv", "year": 2017, "abstract": "abs", "citationCount": 0,
        "journal": None, "publicationDate": "2017-06-12", "arxivId": "1706.03762",
    })
    agent._enrich_and_register_paper(
        {"title": "Attention Is All You Need", "year": 2017, "reason": "r"},
        {"section": "Intro"}, registry, "2024-10",
    )
    bib = agent._generate_bibtex(list(registry.values()))
    assert bib.startswith("@article{")
    assert "journal={arXiv preprint arXiv:1706.03762}" in bib

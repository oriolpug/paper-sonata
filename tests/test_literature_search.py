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
        monkeypatch.setattr(lim, "_next_time", 0.0)
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


AUTOCOMPLETE = {"results": [
    {"id": "https://openalex.org/W1", "display_name": "Attention Is All You Need", "cited_by_count": 150000},
    {"id": "https://openalex.org/W9", "display_name": "Something Else Entirely", "cited_by_count": 5},
]}


def test_fallback_to_openalex_then_cooldown_skips_s2(monkeypatch):
    calls = route(monkeypatch, {
        "semanticscholar": lambda p: _Rsp(429),
        "autocomplete": lambda p: _Rsp(200, AUTOCOMPLETE),
        "openalex.org/works/W": lambda p: _Rsp(200, OPENALEX_WORK),
        "openalex.org/works": lambda p: _Rsp(200, {"results": [OPENALEX_WORK]}),
    })
    hit = su.title_search("Attention Is All You Need", 2017, "2024-10")
    assert hit["source"] == "openalex"
    assert hit["venue"] == "arXiv" and hit["arxivId"] == "1706.03762"
    assert hit["abstract"] == "The dominant models"
    assert hit["authors"] == [{"name": "Ashish Vaswani"}]
    # free path: autocomplete + one per-work fetch (non-matching hit not fetched)
    assert [k for k, _ in calls if "openalex" in k or "autocomplete" in k] == [
        "autocomplete", "openalex.org/works/W"]
    s2_calls = sum(1 for k, _ in calls if k == "semanticscholar")

    su._paper_cache.clear()
    su.keyword_search("transformers attention")
    # S2 is in cooldown, so it is not contacted again
    assert sum(1 for k, _ in calls if k == "semanticscholar") == s2_calls
    assert calls[-1][0] == "openalex.org/works"
    assert calls[-1][1]["filter"].startswith("title_and_abstract.search:")


def test_prefers_published_version_and_requires_abstract(monkeypatch):
    published = dict(OPENALEX_WORK, cited_by_count=10, abstract_inverted_index=None,
                     primary_location={"landing_page_url": "https://doi.org/x",
                                       "source": {"display_name": "NeurIPS", "type": "conference"}})
    works = {"W1": OPENALEX_WORK, "W2": published}
    route(monkeypatch, {
        "semanticscholar": lambda p: _Rsp(429),
        "autocomplete": lambda p: _Rsp(200, {"results": [
            {"id": "https://openalex.org/W1", "display_name": "Attention Is All You Need", "cited_by_count": 150000},
            {"id": "https://openalex.org/W2", "display_name": "Attention is all you need", "cited_by_count": 10},
        ]}),
        "openalex.org/works/W": lambda p: None,
    })
    import utils.scholar_utils as mod
    monkeypatch.setattr(mod, "_openalex_get_work", lambda wid: mod._openalex_normalize(works[wid.rsplit("/", 1)[-1]]))

    hit = su.title_search("Attention Is All You Need", 2017, "2024-10")
    assert hit["venue"] == "NeurIPS"  # published beats the more-cited preprint

    su._paper_cache.clear()
    hit = su.title_search("Attention Is All You Need", 2017, "2024-10", require_abstract=True)
    assert hit["venue"] == "arXiv" and hit["abstract"]  # published one has no abstract


def test_long_retry_after_falls_back_immediately(monkeypatch):
    calls = route(monkeypatch, {
        "semanticscholar": lambda p: _Rsp(404),
        "autocomplete": lambda p: _Rsp(429, {"retryAfter": 35}),
        "arxiv": lambda p: _Rsp(200, content=ARXIV_XML),
    })
    hit = su.title_search("Deep Residual Learning for Image Recognition", 2016, "2024-10")
    assert hit["source"] == "arxiv"
    assert sum(1 for k, _ in calls if k == "autocomplete") == 1  # no retries on 35s wait
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
        "autocomplete": lambda p: _Rsp(200, {"results": []}),
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


def _fake_llm(outputs, prompts):
    def call(model_name, contents, generation_configs=None, **kwargs):
        prompts.append((model_name, contents[0]))
        out = outputs.pop(0)
        return {"raw_response": json.dumps(out), "parsed_response": out}
    return call


def test_db_discovery_flow_scores_relevance(monkeypatch, tmp_path):
    from methods.agents import literature_review_agent as lra

    resnet = su._s2_normalize(RESNET_S2)
    offtopic = dict(su._s2_normalize(RESNET_S2), title="The Year in Cardiovascular Medicine",
                    citationCount=5000)
    known = dict(su._s2_normalize(RESNET_S2), title="Identity Mappings in Deep Residual Networks")
    searched = []
    monkeypatch.setattr(lra, "keyword_search",
                        lambda q, limit, cutoff_date: searched.append(q) or [resnet, offtopic])
    monkeypatch.setattr(lra, "title_search", lambda t, y, c, **kw: known if "Identity" in t else None)
    monkeypatch.setattr(lra, "title_candidates", lambda *a, **k: [])
    prompts = []
    monkeypatch.setattr(lra, "call_llm_with_contents", _fake_llm([
        {"queries": ["Find highly cited papers on residual networks", "deep residual learning"],
         "known_titles": ["Identity Mappings in Deep Residual Networks"]},
        {"scores": [{"index": 0, "score": 3, "reason": "core"},
                    {"index": 1, "score": 3, "reason": "backbone"},
                    {"index": 2, "score": 0, "reason": "other field"}]},
    ], prompts))
    agent = _agent(tmp_path, model_name="ollama_chat/qwen3:8b", fast_model_name="ollama_chat/qwen3:8b")
    (tmp_path / "a").write_text("We study image classification with very deep networks.")
    agent.context_files = agent._load_context_files()
    cands = agent._discover_candidates(
        {"section": "Intro", "focus": "residual nets", "context": "c", "search_type": "exploration"},
        {"intro_related_work_plan": {"introduction_strategy": {"problem_gap_hypothesis": "deep nets degrade"}}},
        "2024-10",
    )
    titles = [c["title"] for c in cands]
    assert "The Year in Cardiovascular Medicine" not in titles  # scored 0
    assert set(titles) == {known["title"], resnet["title"]}
    assert searched == ["residual networks", "deep residual learning"]  # filler stripped
    assert "image classification with very deep networks" in prompts[0][1]  # project brief
    assert "[2] The Year in Cardiovascular Medicine" in prompts[1][1]
    assert all(m == "ollama_chat/qwen3:8b" for m, _ in prompts)


def test_targeted_uses_quoted_name_and_keeps_one(monkeypatch, tmp_path):
    from methods.agents import literature_review_agent as lra

    sam = dict(su._s2_normalize(RESNET_S2), title="Segment Anything")
    other = dict(su._s2_normalize(RESNET_S2), title="SAM 2: Segment Anything in Images and Videos")
    searched, looked_up = [], []
    monkeypatch.setattr(lra, "keyword_search",
                        lambda q, limit, cutoff_date: searched.append(q) or [other])
    # The original paper is only found by the most-cited title lookup.
    monkeypatch.setattr(lra, "title_candidates",
                        lambda q, limit, cutoff_date: looked_up.append(q) or [sam])
    monkeypatch.setattr(lra, "title_search", lambda *a, **k: None)
    monkeypatch.setattr(lra, "call_llm_with_contents", _fake_llm([
        {"queries": ["segment anything promptable segmentation"], "known_titles": []},
        {"scores": [{"index": 0, "score": 3}, {"index": 1, "score": 0}]},
    ], []))
    agent = _agent(tmp_path, fast_model_name="ollama_chat/qwen3:8b")
    cands = agent._discover_candidates(
        {"section": "RW", "focus": "research paper introducing 'Segment Anything Model (SAM)'",
         "context": "c", "search_type": "targeted"},
        {}, "2024-10",
    )
    assert searched[0] == "Segment Anything Model SAM"
    assert looked_up[0] == "Segment Anything Model SAM"
    assert [c["title"] for c in cands] == ["Segment Anything"]


def test_clean_search_query():
    from methods.agents.literature_review_agent import clean_search_query as c

    assert c("Find highly cited papers establishing the real-world impact of temporal "
             "inconsistency in video segmentation") == "temporal inconsistency video segmentation"
    assert c("Identify foundational papers establishing causal video segmentation") == "causal video segmentation"
    assert len(c("a b c d e f g h i j k l m n o p").split()) <= 8


def test_parse_relevance_scores():
    from methods.agents.literature_review_agent import parse_relevance_scores as p

    assert p({"scores": [{"index": "[1]", "score": "2", "reason": "x"}, {"oops": 1}]}) == [
        {"index": 1, "score": 2.0, "reason": "x"}]
    assert [i["index"] for i in p({"0": 3, "1": 0})] == [0, 1]
    assert p("garbage") == []


def test_google_backend_falls_back_for_non_gemini(tmp_path):
    assert _agent(tmp_path, fast_model_name="openai/gpt-5", search_backend="google").search_backend == "s2"
    assert _agent(tmp_path, fast_model_name="gemini-3-flash-preview").search_backend == "google"


def test_preprint_bibtex(monkeypatch, tmp_path):
    from methods.agents import literature_review_agent as lra

    agent = _agent(tmp_path)
    registry = {}
    monkeypatch.setattr(lra, "title_search", lambda t, y, c, **kw: {
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


def test_waits_when_all_sources_cooling_down(monkeypatch):
    """When every source is rate-limited, wait for the first to recover
    instead of reporting 'no results'."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(su.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(su.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setenv("PAPER_SEARCH_SOURCES", "s2,arxiv")
    su._cooldown_until.update({"s2": 1300.0, "arxiv": 1020.0})
    calls = route(monkeypatch, {"arxiv": lambda p: _Rsp(200, content=ARXIV_XML)})
    hit = su.title_search("Deep Residual Learning for Image Recognition", 2016, "2024-10")
    assert hit["source"] == "arxiv"
    assert 1020.0 <= clock["t"] < 1030.0  # waited ~20s for arXiv, not 300s for S2
    assert [k for k, _ in calls] == ["arxiv"]


def test_gives_up_after_max_wait(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(su.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(su.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(su, "SEARCH_MAX_WAIT", 60.0)
    monkeypatch.setenv("PAPER_SEARCH_SOURCES", "s2")
    su._cooldown_until.update({"s2": 1300.0})
    route(monkeypatch, {})
    assert su.keyword_search("anything") == []
    assert clock["t"] == 1000.0  # did not sleep past the budget


def test_timeout_uses_short_cooldown(monkeypatch):
    def boom(*a, **k):
        raise su.requests.ReadTimeout("slow")

    monkeypatch.setattr(su.requests, "get", boom)
    monkeypatch.setenv("PAPER_SEARCH_SOURCES", "arxiv")
    monkeypatch.setattr(su, "SEARCH_MAX_WAIT", 0.0)
    assert su.keyword_search("residual networks") == []
    remaining = su._cooldown_until["arxiv"] - su.time.monotonic()
    assert 0 < remaining <= su.NETWORK_ERROR_COOLDOWN


def test_title_lookup_variants():
    from methods.agents.literature_review_agent import title_lookup_variants as v

    assert v(["segment anything model", "Ref-AVS dataset"]) == [
        "segment anything model", "segment anything", "Ref-AVS dataset", "Ref-AVS"]
    assert v(["model"]) == ["model"]

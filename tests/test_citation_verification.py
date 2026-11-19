"""Background verification (A) and LLM-assisted recovery (B) in the literature agent."""
import json
import threading

import pytest

from methods.agents import literature_review_agent as lra


def make_agent(tmp_path, **kw):
    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text("x")
    kw.setdefault("fast_model_name", "ollama_chat/qwen3:8b")
    kw.setdefault("model_name", "ollama_chat/qwen3:8b")
    return lra.HybridLiteratureAgent(
        idea_path=str(tmp_path / "a"), experimental_log_path=str(tmp_path / "b"),
        latex_template_path=str(tmp_path / "c"), conference_guidelines_path=str(tmp_path / "d"),
        output_dir=str(tmp_path / "out"), **kw,
    )


def record(title, year=2020, venue="NeurIPS", abstract="An abstract."):
    return {
        "title": title, "authors": [{"name": "Ada Lovelace"}], "venue": venue,
        "year": year, "abstract": abstract, "citationCount": 10, "journal": None,
        "publicationDate": f"{year}-01-01", "arxivId": None, "doi": None, "source": "test",
    }


TASK = {"section": "Intro", "focus": "f", "context": "c", "search_type": "exploration"}


def test_verification_runs_while_discovery_continues(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    verified_early = threading.Event()
    slow_saw_verification = {}

    def fake_process(task, outline, cutoff):
        if task["focus"] == "fast":
            return [{"title": "Fast Paper", "year": 2020, "reason": "r"}]
        # The slow task only finishes once the fast task's candidate has been
        # verified, which proves verification overlaps with discovery.
        slow_saw_verification["value"] = verified_early.wait(timeout=5)
        return [{"title": "Slow Paper", "year": 2020, "reason": "r"}]

    def fake_title_search(title, year, cutoff, require_abstract=False):
        if title == "Fast Paper":
            verified_early.set()
        return record(title)

    monkeypatch.setattr(agent, "_process_search_task", fake_process)
    monkeypatch.setattr(lra, "title_search", fake_title_search)
    registry = {}
    agent._execute_tasks_in_parallel(
        [dict(TASK, focus="fast"), dict(TASK, focus="slow")], {}, registry, set(), "2024-10"
    )
    assert slow_saw_verification["value"] is True
    assert {p.title for p in registry.values()} == {"Fast Paper", "Slow Paper"}


def test_duplicate_candidates_verified_once(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    calls = []
    monkeypatch.setattr(
        agent, "_process_search_task",
        lambda task, o, c: [{"title": "Same Paper", "year": 2020, "reason": "r"}],
    )
    monkeypatch.setattr(
        lra, "title_search",
        lambda t, y, c, require_abstract=False: calls.append(t) or record(t),
    )
    registry = {}
    agent._execute_tasks_in_parallel(
        [dict(TASK, focus=str(i)) for i in range(5)], {}, registry, set(), "2024-10"
    )
    assert len(registry) == 1
    assert calls == ["Same Paper"]


def fake_llm(outputs, prompts):
    def call(model_name, contents, generation_configs=None, **kwargs):
        prompts.append(contents[0])
        out = outputs.pop(0)
        return {"raw_response": json.dumps(out), "parsed_response": out}
    return call


def test_recovery_finds_corrected_title(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    real = record("Attention Is All You Need", 2017)

    def fake_title_search(title, year, cutoff, require_abstract=False):
        return real if title == "Attention Is All You Need" else None

    prompts = []
    monkeypatch.setattr(lra, "title_search", fake_title_search)
    monkeypatch.setattr(lra, "keyword_search", lambda q, limit, cutoff_date: [])
    monkeypatch.setattr(lra, "call_llm_with_contents", fake_llm([
        {"queries": ["Attention Is All You Need", "transformer self-attention"]},
        {"match_index": 0, "explanation": "same paper"},
    ], prompts))

    registry = {}
    agent._enrich_and_register_paper(
        {"title": "Attention is all we need (Transformers)", "year": 2017, "reason": "r"},
        TASK, registry, "2024-10",
    )
    assert [p.title for p in registry.values()] == ["Attention Is All You Need"]
    assert agent.verification_log["recovered"][0]["candidate_title"].startswith("Attention is all we need")
    assert "Attention Is All You Need (2017" in prompts[1]


def test_recovery_rejects_hallucination(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    prompts = []
    monkeypatch.setattr(lra, "title_search", lambda *a, **k: None)
    monkeypatch.setattr(
        lra, "keyword_search", lambda q, limit, cutoff_date: [record("Unrelated Work")]
    )
    monkeypatch.setattr(lra, "call_llm_with_contents", fake_llm([
        {"queries": ["quantum blockchain diffusion"]},
        {"match_index": None, "explanation": "no match"},
    ], prompts))

    registry = {}
    agent._enrich_and_register_paper(
        {"title": "Quantum Blockchain Diffusion Nets", "year": 2023, "reason": "r"},
        TASK, registry, "2024-10",
    )
    assert registry == {}
    assert agent.verification_log["rejected"][0]["reason"] == "nonexistent"
    agent._save_verification_log(0)
    saved = json.loads((tmp_path / "out" / "candidate_verification.json").read_text())
    assert saved["rejected"][0]["title"] == "Quantum Blockchain Diffusion Nets"
    assert saved["recovery_attempts"] == 1


def test_recovery_budget(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, max_citation_recoveries=0)
    monkeypatch.setattr(lra, "title_search", lambda *a, **k: None)
    monkeypatch.setattr(
        lra, "call_llm_with_contents",
        lambda **k: pytest.fail("recovery must not call the LLM when disabled"),
    )
    agent._enrich_and_register_paper({"title": "Missing", "year": 2020}, TASK, {}, "2024-10")
    assert agent.verification_log["rejected"][0]["reason"] == "not_found"


def test_recovery_llm_failure_is_contained(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    monkeypatch.setattr(lra, "title_search", lambda *a, **k: None)

    def boom(**kwargs):
        raise RuntimeError("model offline")

    monkeypatch.setattr(lra, "call_llm_with_contents", boom)
    agent._enrich_and_register_paper({"title": "Missing", "year": 2020}, TASK, {}, "2024-10")
    assert agent.verification_log["rejected"][0]["reason"] == "not_found"

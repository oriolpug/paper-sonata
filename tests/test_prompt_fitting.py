"""Fitting prompts into small context windows (local models)."""
import json

import pytest

from utils import llm, prompt_fitting as pf
from utils.llm_parsing import parse_latex_document


def _render_factory(n_items=10, abstract_len=1000):
    papers = [{"key": f"p{i}", "abstract": "word " * (abstract_len // 5)} for i in range(n_items)]

    def render(abstract_chars, n):
        return "HEADER " + " ".join(
            p["key"] + ":" + pf.truncate_abstract(p["abstract"], abstract_chars) for p in papers[:n]
        )
    return render


def test_no_budget_means_unchanged(monkeypatch):
    monkeypatch.setattr(pf, "prompt_budget", lambda m, r: None)
    text, level, n = pf.fit_papers("gemini/x", _render_factory(), 10, 8192)
    assert level is None and n == 10


def test_shortens_abstracts_then_drops_papers(monkeypatch):
    monkeypatch.setattr(pf, "count_tokens", lambda m, t: len(t))
    render = _render_factory()
    monkeypatch.setattr(pf, "prompt_budget", lambda m, r: 4000)
    _, level, n = pf.fit_papers("ollama_chat/x", render, 10, 0)
    assert level == 250 and n == 10
    monkeypatch.setattr(pf, "prompt_budget", lambda m, r: 40)
    text, level, n = pf.fit_papers("ollama_chat/x", render, 10, 0)
    assert level == 0 and n < 10 and len(text) <= 40
    assert pf.describe_fit(10, level, n).startswith("Prompt shrunk")
    assert pf.describe_fit(10, None, 10) is None


def test_truncate_abstract():
    assert pf.truncate_abstract("one two three four", 9) == "one two..."
    assert pf.truncate_abstract("short", 100) == "short"
    assert pf.truncate_abstract("anything", 0) == ""


def test_context_window_and_budget(monkeypatch):
    monkeypatch.delenv("PAPER_CONTEXT_WINDOW", raising=False)
    monkeypatch.setenv("PAPER_OLLAMA_NUM_CTX", "32768")
    assert llm.context_window("ollama_chat/qwen3:8b") == 32768
    assert llm.prompt_budget("ollama_chat/qwen3:8b", 8192) == 24576
    assert llm.prompt_budget("gemini/gemini-3.1-pro-preview", 8192) is None  # 1M window
    monkeypatch.setenv("PAPER_CONTEXT_WINDOW", "16000")
    assert llm.context_window("openai/gpt-5") == 16000


def test_ollama_overflow_fails_loudly(monkeypatch):
    monkeypatch.setenv("PAPER_OLLAMA_NUM_CTX", "100")
    monkeypatch.setattr(llm, "count_tokens", lambda m, t: 500)
    monkeypatch.setattr(llm.litellm, "completion", lambda **k: pytest.fail("must not be sent"))
    with pytest.raises(llm.PromptTooLongError, match="truncate"):
        llm.call_llm(["x" * 2000], "ollama_chat/qwen3:8b")


def test_parse_latex_document():
    assert parse_latex_document("```latex\n\\begin{document}x\\end{document}\n```") == "\\begin{document}x\\end{document}"
    assert parse_latex_document("### A markdown summary of papers") is None


def test_synthesis_shrinks_to_fit(monkeypatch, tmp_path):
    from methods.agents import literature_review_agent as lra

    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text("x")
    agent = lra.HybridLiteratureAgent(
        idea_path=str(tmp_path / "a"), experimental_log_path=str(tmp_path / "b"),
        latex_template_path=str(tmp_path / "c"), conference_guidelines_path=str(tmp_path / "d"),
        output_dir=str(tmp_path / "out"), model_name="ollama_chat/qwen3:8b",
        fast_model_name="ollama_chat/qwen3:8b")
    papers = [lra.PaperData(citation_key=f"k{i}", title=f"T{i}", authors=["A"], venue="V", year=2020,
                            abstract="word " * 300, citation_count=i,
                            found_in_section="Introduction" if i % 2 else "3. Method - x", reason="r")
              for i in range(40)]
    seen = {}

    def fake_call(model_name, contents, generation_configs=None, result_parsing_func=None, **kw):
        seen["context"] = json.loads(contents[1])
        seen["system"] = generation_configs["system_instruction"]
        return {"parsed_response": "\\begin{document}\\end{document}"}

    monkeypatch.setattr(lra, "call_llm_with_contents", fake_call)
    monkeypatch.setattr(pf, "count_tokens", lambda m, t: len(t) // 4)
    monkeypatch.setattr(pf, "prompt_budget", lambda m, r: 3000)
    agent._synthesize_content({}, papers, "2024-11")
    kept = seen["context"]["collected_papers"]
    assert 0 < len(kept) < 40
    # Introduction papers are kept first
    assert all(p["found_in_section"] == "Introduction" for p in kept[: min(len(kept), 20)])
    assert seen["context"]["citation_checklist"] == [p["citation_key"] for p in kept]
    assert f"{len(kept)}" in seen["system"]

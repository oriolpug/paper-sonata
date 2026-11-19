import base64
import io

import pymupdf  # type: ignore
import pytest

from utils import llm
from utils.llm_backend_utils import (
    call_llm_with_contents,
    call_llm_with_images,
    call_llm_with_pdf,
    call_llm_with_text_prompt,
)
from utils.model_defaults import resolve_model_names

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "GEMINI_API_KEY",
        "VERTEX_AI_PROJECT",
        "VERTEX_AI_LOCATION",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "PAPER_LLM_SUPPORTS_VISION",
        "PAPER_LLM_SUPPORTS_PDF",
        "PAPER_FAST_MODEL",
        "PAPER_IMAGE_MODEL",
        "PAPER_REFLECTION_MODEL",
        "PAPER_PLOTTING_MODEL",
        "PAPER_MAX_OUTPUT_TOKENS",
        "PAPER_OLLAMA_THINK",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PaperOrchestra. " * 20)
    doc.save(str(path))
    return str(path)


class FakeCompletion:
    """Records litellm.completion calls and returns canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item

        class Msg:
            content = item

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


@pytest.fixture
def fake(monkeypatch):
    def install(*responses):
        f = FakeCompletion(responses)
        monkeypatch.setattr(llm.litellm, "completion", f)
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        return f

    return install


# --- model names -----------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("gemini-3.1-pro-preview", "gemini/gemini-3.1-pro-preview"),
        ("gpt-5", "openai/gpt-5"),
        ("o3-mini", "openai/o3-mini"),
        ("claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"),
        ("ollama_chat/qwen3:8b", "ollama_chat/qwen3:8b"),
        ("hosted_vllm/Qwen/Qwen3-32B", "hosted_vllm/Qwen/Qwen3-32B"),
    ],
)
def test_normalize_model_name(name, expected):
    assert llm.normalize_model_name(name) == expected


def test_bare_gemini_prefers_vertex_when_configured(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_PROJECT", "p")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")
    assert llm.normalize_model_name("gemini-3-flash-preview") == "vertex_ai/gemini-3-flash-preview"
    assert llm._completion_kwargs("vertex_ai/gemini-3-flash-preview") == {
        "vertex_project": "p",
        "vertex_location": "us-central1",
    }


def test_is_gemini_model():
    assert llm.is_gemini_model("gemini-3.1-pro-preview")
    assert llm.is_gemini_model("vertex_ai/gemini-3-flash-preview")
    assert not llm.is_gemini_model("ollama_chat/gemma3")
    assert not llm.is_gemini_model("openai/gpt-5")


def test_model_file_tag():
    assert llm.model_file_tag("gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"
    assert llm.model_file_tag("ollama_chat/qwen3:8b") == "ollama_chat_qwen3_8b"


def test_resolve_model_names_follows_non_gemini_writer():
    m = resolve_model_names(writer="ollama_chat/qwen3:8b")
    assert m["fast"] == m["reflection"] == m["plotting"] == "ollama_chat/qwen3:8b"
    assert m["image"] == "none"


def test_resolve_model_names_gemini_defaults():
    m = resolve_model_names(writer="gemini-3.1-pro-preview")
    assert m["fast"] == "gemini-3-flash-preview"
    assert m["image"] == "gemini-3-pro-image-preview"
    assert resolve_model_names(writer="gpt-5", image="openai/gpt-image-1")["image"] == "openai/gpt-image-1"


# --- message building ------------------------------------------------------


def test_text_only_collapses_to_string():
    msgs = llm.build_messages(["a", llm.text("b")], "openai/gpt-5", system_prompt="sys")
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a\n\nb"},
    ]


def test_image_is_sent_as_data_uri_with_sniffed_mime():
    content = llm.build_user_content(
        ["look", llm.image(PNG_1PX, mime_type="image/jpeg")], "openai/gpt-5"
    )
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_images_dropped_for_non_vision_model(monkeypatch):
    monkeypatch.setenv("PAPER_LLM_SUPPORTS_VISION", "false")
    content = llm.build_user_content(["look", llm.image(PNG_1PX)], "ollama_chat/llama3")
    assert isinstance(content, str)
    assert "omitted" in content


def test_pdf_native_vs_text_fallback(sample_pdf, monkeypatch):
    native = llm.build_user_content([llm.pdf(sample_pdf)], "gemini/gemini-3.1-pro-preview")
    assert native[0]["type"] == "file"
    assert native[0]["file"]["file_data"].startswith("data:application/pdf;base64,")

    fallback = llm.build_user_content([llm.pdf(sample_pdf)], "ollama_chat/qwen3:8b")
    assert isinstance(fallback, str)
    assert "Hello PaperOrchestra" in fallback


# --- call_llm --------------------------------------------------------------


def test_call_llm_parses_json_and_maps_configs(fake):
    f = fake('Sure!\n```json\n{"a": 1}\n```')
    out = call_llm_with_text_prompt(
        "hi",
        "gpt-5",
        generation_configs={"system_instruction": "be terse", "temperature": 0.2},
    )
    assert out["parsed_response"] == {"a": 1}
    call = f.calls[0]
    assert call["model"] == "openai/gpt-5"
    assert call["temperature"] == 0.2
    assert call["messages"][0] == {"role": "system", "content": "be terse"}


def test_call_llm_strips_think_blocks(fake):
    fake('<think>maybe {"wrong": true}</think>\n{"right": true}')
    out = call_llm_with_text_prompt("hi", "ollama_chat/qwen3:8b")
    assert out["parsed_response"] == {"right": True}
    assert "<think>" not in out["raw_response"]


def test_call_llm_retries_on_unparseable_output(fake):
    f = fake("no json here", '{"ok": 1}')
    out = call_llm_with_text_prompt("hi", "openai/gpt-5")
    assert out["parsed_response"] == {"ok": 1}
    assert len(f.calls) == 2


def test_call_llm_no_retry_on_context_window(fake):
    err = llm.litellm.ContextWindowExceededError(
        message="too long", model="openai/gpt-5", llm_provider="openai"
    )
    f = fake(err, '{"ok": 1}')
    with pytest.raises(llm.litellm.ContextWindowExceededError):
        call_llm_with_text_prompt("hi", "openai/gpt-5")
    assert len(f.calls) == 1


def test_call_llm_drops_images_when_rejected(fake, tmp_path):
    img = tmp_path / "fig.png"
    img.write_bytes(PNG_1PX)
    err = llm.litellm.BadRequestError(
        message="model does not support image input", model="x", llm_provider="ollama"
    )
    f = fake(err, '{"ok": 1}')
    out = call_llm_with_images("describe", [str(img)], "ollama_chat/unknown-model")
    assert out["parsed_response"] == {"ok": 1}
    assert isinstance(f.calls[0]["messages"][-1]["content"], list)
    assert isinstance(f.calls[1]["messages"][-1]["content"], str)


def test_call_llm_with_pdf_text_fallback(fake, sample_pdf):
    f = fake('{"score": 3}')
    out = call_llm_with_pdf(sample_pdf, "rate it", "ollama_chat/qwen3:8b", system_instruction="sys")
    assert out["parsed_response"] == {"score": 3}
    user = f.calls[0]["messages"][-1]["content"]
    assert "Hello PaperOrchestra" in user and "rate it" in user


def test_ollama_gets_num_ctx(fake):
    f = fake("plain text")
    call_llm_with_contents(["hi"], "ollama_chat/qwen3:8b", return_json=False)
    assert f.calls[0]["num_ctx"] == 32768
    assert f.calls[0]["think"] is False


def test_ollama_think_opt_in(fake, monkeypatch):
    monkeypatch.setenv("PAPER_OLLAMA_THINK", "true")
    f = fake("plain text")
    call_llm_with_contents(["hi"], "ollama_chat/qwen3:8b", return_json=False)
    assert f.calls[0]["think"] is True
    f2 = fake("plain")
    call_llm_with_contents(["hi"], "ollama_chat/gpt-oss:20b", return_json=False)
    assert "think" not in f2.calls[0]


def test_tools_passed_through(fake):
    f = fake('{"candidates": []}')
    call_llm_with_contents(
        ["find"], "gemini/gemini-3-flash-preview",
        generation_configs={"tools": llm.web_search_tools("gemini/gemini-3-flash-preview")},
    )
    assert f.calls[0]["tools"] == [{"googleSearch": {}}]
    assert llm.web_search_tools("ollama_chat/qwen3") is None

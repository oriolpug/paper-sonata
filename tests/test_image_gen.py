import base64

from utils import image_gen


class _Item:
    def __init__(self, b64):
        self.b64_json = b64
        self.url = None


class _Resp:
    def __init__(self, b64):
        self.data = [_Item(b64)]


def test_disabled_image_model_returns_none(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(image_gen.litellm, "image_generation", boom)
    assert image_gen.generate_image("none", "a diagram") is None
    assert image_gen.generate_image(None, "a diagram") is None


def test_gemini_image_params_and_save(monkeypatch, tmp_path):
    calls = []

    def fake(prompt, model, **kwargs):
        calls.append({"prompt": prompt, "model": model, **kwargs})
        return _Resp(base64.b64encode(b"img").decode())

    monkeypatch.setattr(image_gen.litellm, "image_generation", fake)
    out_path = tmp_path / "fig.png"
    b64 = image_gen.generate_image(
        "gemini-3-pro-image-preview", "draw", aspect_ratio="16:9",
        system_prompt="style", save_path=str(out_path),
    )
    assert base64.b64decode(b64) == b"img"
    assert out_path.read_bytes() == b"img"
    assert calls[0]["model"] == "gemini/gemini-3-pro-image-preview"
    assert calls[0]["imageConfig"] == {"aspectRatio": "16:9"}
    assert calls[0]["prompt"].startswith("style")


def test_openai_image_size(monkeypatch):
    calls = []

    def fake(prompt, model, **kwargs):
        calls.append(kwargs)
        return _Resp(base64.b64encode(b"x").decode())

    monkeypatch.setattr(image_gen.litellm, "image_generation", fake)
    image_gen.generate_image("openai/gpt-image-1", "draw", aspect_ratio="9:16")
    assert calls[0]["size"] == "1024x1536"


def test_plotting_agent_skips_diagram_when_disabled(tmp_path):
    from methods.agents.plotting_agent import process_single_figure

    res = process_single_figure(
        {"figure_id": "fig1", "plot_type": "diagram"},
        str(tmp_path),
        model_name="openai/gpt-5",
        image_model_name="none",
    )
    assert res is None

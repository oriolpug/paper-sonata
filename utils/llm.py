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

"""Provider-agnostic LLM access built on LiteLLM.

Any model supported by LiteLLM can be used: hosted APIs (Gemini, OpenAI,
Anthropic, OpenRouter, ...) and open-source models served by Ollama, vLLM,
LM Studio or any OpenAI-compatible server. Model names follow LiteLLM's
`<provider>/<model>` convention; see `normalize_model_name` for legacy names.
"""

import base64
import mimetypes
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import litellm  # type: ignore
from dotenv import load_dotenv  # type: ignore

from utils.llm_parsing import parse_json

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Silently drop parameters a provider does not support (e.g. temperature on
# OpenAI reasoning models) instead of failing the request.
litellm.drop_params = True
litellm.suppress_debug_info = True

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


@dataclass
class Part:
    kind: str  # "text" | "image" | "pdf"
    text: Optional[str] = None
    data: Optional[bytes] = None
    mime_type: Optional[str] = None
    path: Optional[str] = None


def text(s: str) -> Part:
    return Part(kind="text", text=s)


def _read(source: Union[str, bytes]) -> tuple[bytes, Optional[str]]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None
    with open(source, "rb") as f:
        return f.read(), source


def _sniff_image_mime(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image(source: Union[str, bytes], mime_type: Optional[str] = None) -> Part:
    data, path = _read(source)
    # Trust the actual bytes over the declared type: some providers (e.g.
    # Anthropic) reject images whose declared media type does not match.
    mime_type = (
        _sniff_image_mime(data)
        or mime_type
        or (mimetypes.guess_type(path)[0] if path else None)
        or "image/png"
    )
    return Part(kind="image", data=data, mime_type=mime_type, path=path)


def pdf(source: Union[str, bytes]) -> Part:
    data, path = _read(source)
    return Part(kind="pdf", data=data, mime_type="application/pdf", path=path)


Content = Union[str, Part]


# ---------------------------------------------------------------------------
# Model names and capabilities
# ---------------------------------------------------------------------------


def _use_vertex() -> bool:
    return bool(
        (os.getenv("VERTEX_AI_PROJECT") or os.getenv("VERTEXAI_PROJECT"))
        and (os.getenv("VERTEX_AI_LOCATION") or os.getenv("VERTEXAI_LOCATION"))
    )


def normalize_model_name(model_name: str) -> str:
    """Maps legacy bare model names to LiteLLM `<provider>/<model>` names.

    Names that already contain a provider prefix are returned unchanged.
    Bare Gemini names go to Vertex AI when VERTEX_AI_PROJECT/LOCATION are set
    (matching the original behaviour), otherwise to the Gemini API.
    """
    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    name = model_name.strip()
    if "/" in name:
        return name
    lower = name.lower()
    if lower.startswith(("gemini", "imagen")):
        return f"vertex_ai/{name}" if _use_vertex() else f"gemini/{name}"
    if lower.startswith(_OPENAI_PREFIXES) or lower.startswith("gpt-image"):
        return f"openai/{name}"
    if lower.startswith("claude"):
        return f"anthropic/{name}"
    return name


def get_provider(model_name: str) -> str:
    name = normalize_model_name(model_name)
    return name.split("/", 1)[0] if "/" in name else ""


def is_gemini_model(model_name: str) -> bool:
    name = normalize_model_name(model_name)
    return get_provider(name) in ("gemini", "vertex_ai") and "gemini" in name.lower()


def is_disabled_model(model_name: Optional[str]) -> bool:
    return model_name is None or str(model_name).strip().lower() in (
        "",
        "none",
        "null",
        "off",
        "false",
    )


def _env_flag(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _model_capability(model_name: str, key: str) -> Optional[bool]:
    """True/False if LiteLLM knows the model, None if the model is unknown."""
    try:
        info = litellm.get_model_info(model_name)
    except Exception:
        return None
    value = info.get(key)
    return None if value is None else bool(value)


def is_chatgpt_plan_model(model_name: str) -> bool:
    """`chatgpt/<model>`: the user's ChatGPT plan via Sign in with ChatGPT.

    Handled by utils.chatgpt_backend (official Responses API), never by
    LiteLLM's own `chatgpt/` provider.
    """
    return str(model_name).strip().startswith("chatgpt/")


def supports_vision(model_name: str) -> Optional[bool]:
    override = _env_flag("PAPER_LLM_SUPPORTS_VISION")
    if override is not None:
        return override
    if is_chatgpt_plan_model(model_name):
        return None  # depends on the plan's model catalog; send and see
    return _model_capability(model_name, "supports_vision")


def supports_pdf_input(model_name: str) -> bool:
    override = _env_flag("PAPER_LLM_SUPPORTS_PDF")
    if override is not None:
        return override
    if is_chatgpt_plan_model(model_name):
        return True  # Responses API accepts inline files for models that support them
    return bool(_model_capability(model_name, "supports_pdf_input"))


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _pdf_to_text(part: Part) -> str:
    from utils.pdf_utils import load_paper

    if part.path:
        return load_paper(part.path, min_size=100)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(part.data)
        tmp_path = f.name
    try:
        return load_paper(tmp_path, min_size=100)
    finally:
        os.remove(tmp_path)


def _data_uri(part: Part) -> str:
    return f"data:{part.mime_type};base64,{base64.b64encode(part.data).decode('utf-8')}"


def build_user_content(
    contents: Iterable[Content],
    model_name: str,
    drop_images: bool = False,
) -> Union[str, List[Dict[str, Any]]]:
    """Converts parts into an OpenAI-style user message content.

    Returns a plain string when only text is present, which keeps maximum
    compatibility with simple OpenAI-compatible servers.
    """
    pdf_native = supports_pdf_input(model_name)
    if not drop_images and supports_vision(model_name) is False:
        drop_images = True

    blocks: List[Dict[str, Any]] = []
    dropped = 0
    for item in contents:
        part = text(item) if isinstance(item, str) else item
        if part.kind == "text":
            blocks.append({"type": "text", "text": part.text or ""})
        elif part.kind == "image":
            if drop_images:
                dropped += 1
                name = os.path.basename(part.path) if part.path else "image"
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Image '{name}' omitted: model does not accept images]",
                    }
                )
            else:
                blocks.append({"type": "image_url", "image_url": {"url": _data_uri(part)}})
        elif part.kind == "pdf":
            if pdf_native:
                block: Dict[str, Any] = {"type": "file", "file": {"file_data": _data_uri(part)}}
                if part.path:
                    block["file"]["filename"] = os.path.basename(part.path)
                blocks.append(block)
            else:
                blocks.append(
                    {
                        "type": "text",
                        "text": "[PDF CONTENT START]\n"
                        + _pdf_to_text(part)
                        + "\n[PDF CONTENT END]",
                    }
                )
        else:
            raise ValueError(f"Unknown content part kind: {part.kind}")

    if dropped:
        print(
            f"Warning: dropped {dropped} image(s) because model '{model_name}' does not "
            "support vision input. Set PAPER_LLM_SUPPORTS_VISION=true to force sending them."
        )

    if all(b["type"] == "text" for b in blocks):
        return "\n\n".join(b["text"] for b in blocks)
    return blocks


def build_messages(
    contents: Iterable[Content],
    model_name: str,
    system_prompt: Optional[str] = None,
    drop_images: bool = False,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": build_user_content(contents, model_name, drop_images=drop_images),
        }
    )
    return messages


def _completion_kwargs(model_name: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    provider = get_provider(model_name)
    if provider == "vertex_ai":
        project = os.getenv("VERTEXAI_PROJECT") or os.getenv("VERTEX_AI_PROJECT")
        location = os.getenv("VERTEXAI_LOCATION") or os.getenv("VERTEX_AI_LOCATION")
        if project:
            kwargs["vertex_project"] = project
        if location:
            kwargs["vertex_location"] = location
    if provider in ("ollama", "ollama_chat"):
        # Ollama's default context window is small and silently truncates the
        # long prompts this pipeline sends.
        kwargs["num_ctx"] = int(os.getenv("PAPER_OLLAMA_NUM_CTX", "32768"))
        # Reasoning models (Qwen3, DeepSeek-R1, ...) "think" by default, which
        # is ~30x slower on local hardware. Off unless PAPER_OLLAMA_THINK=true.
        # gpt-oss only accepts effort levels, so leave it alone.
        think = _env_flag("PAPER_OLLAMA_THINK")
        if not model_name.split("/", 1)[-1].startswith("gpt-oss"):
            kwargs["think"] = bool(think)
    max_tokens = os.getenv("PAPER_MAX_OUTPUT_TOKENS")
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    timeout = os.getenv("PAPER_LLM_TIMEOUT")
    if timeout:
        kwargs["timeout"] = float(timeout)
    return kwargs


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _extract_text(response: Any) -> Optional[str]:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        return None
    if isinstance(content, list):
        content = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    if content is None:
        return None
    return _THINK_RE.sub("", content)


def _is_image_rejection(error: Exception) -> bool:
    msg = str(error).lower()
    return any(k in msg for k in ("image", "vision", "multimodal", "image_url"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def call_llm(
    contents: Union[Content, Iterable[Content]],
    model_name: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    result_parsing_func: Optional[Callable] = parse_json,
    check_parsed_response_not_none: bool = True,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    **extra_params: Any,
) -> Dict[str, Any]:
    """Calls any LiteLLM-supported model with retries.

    Returns {"raw_response": str, "parsed_response": Any}.
    """
    if isinstance(contents, (str, Part)):
        contents = [contents]
    contents = list(contents)
    model = normalize_model_name(model_name)
    parser = result_parsing_func or (lambda x: x)

    params: Dict[str, Any] = _completion_kwargs(model)
    if temperature is not None:
        params["temperature"] = temperature
    if tools:
        params["tools"] = tools
    params.update(extra_params)

    raw_response = None
    parsed_response = None
    drop_images = False
    messages = build_messages(contents, model, system_prompt)

    chatgpt_plan = is_chatgpt_plan_model(model)
    if chatgpt_plan:
        from utils import chatgpt_auth, chatgpt_backend

        non_retryable: tuple = (
            chatgpt_backend.ChatGPTPlanError,
            chatgpt_auth.ChatGPTAuthError,
        )
        # temperature / tools / max_tokens are not supported with ChatGPT plan
        # usage and are dropped (like litellm.drop_params for other providers).
    else:
        non_retryable = ()

    for attempt in range(max_retries):
        try:
            if chatgpt_plan:
                raw_response = _THINK_RE.sub(
                    "",
                    chatgpt_backend.complete(model, messages, timeout=params.get("timeout")),
                )
            else:
                response = litellm.completion(model=model, messages=messages, **params)
                raw_response = _extract_text(response)
            if raw_response is None:
                raise ValueError(f"Empty response from model '{model}'")

            parsed_response = parser(raw_response)
            if check_parsed_response_not_none:
                assert (
                    parsed_response is not None
                ), f"Parsed response is None, Here is the raw response:\n{raw_response}"

            return {"raw_response": raw_response, "parsed_response": parsed_response}

        except (
            litellm.ContextWindowExceededError,
            litellm.AuthenticationError,
            litellm.NotFoundError,
            *non_retryable,
        ) as e:
            print(f"Error: LLM call (model={model}) failed with non-retryable error: {e}")
            raise
        except litellm.BadRequestError as e:
            has_images = any(isinstance(c, Part) and c.kind == "image" for c in contents)
            if has_images and not drop_images and _is_image_rejection(e):
                print(
                    f"Warning: model '{model}' rejected image input; retrying without images."
                )
                drop_images = True
                messages = build_messages(contents, model, system_prompt, drop_images=True)
                continue
            print(f"Error: LLM call (model={model}) rejected as bad request: {e}")
            raise
        except Exception as e:
            print(
                f"Warning: LLM call (model={model}) failed on attempt {attempt + 1} of {max_retries}. Error: {e}"
            )
            if attempt + 1 == max_retries:
                print(f"Error: All {max_retries} retry attempts failed.")
                raise

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    print("Error: Exited retry loop unexpectedly.")
    return {"raw_response": raw_response, "parsed_response": parsed_response}


def web_search_tools(model_name: str) -> Optional[List[Dict[str, Any]]]:
    """Native web-search tool for models that support it (currently Gemini)."""
    if is_gemini_model(model_name):
        return [{"googleSearch": {}}]
    return None


def model_file_tag(model_name: str) -> str:
    """Filesystem-safe version of a model name (for result file names)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name)).strip("_")

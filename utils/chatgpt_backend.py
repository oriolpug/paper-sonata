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

"""Inference on the user's ChatGPT plan via Sign in with ChatGPT.

Requests go to the official Responses API (`POST https://api.openai.com/v1/responses`)
with the OAuth access token from `utils.chatgpt_auth`. Plan usage requires
`store: false` and `stream: true`, does not accept `temperature` or
`max_output_tokens`, and does not support image generation.
"""

import threading
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, APIStatusError, OpenAI  # type: ignore

from utils import chatgpt_auth
from utils.chatgpt_auth import USAGE_SETTINGS_URL

PROVIDER = "chatgpt"


class ChatGPTPlanError(Exception):
    """A ChatGPT plan request failed and should not be retried as-is."""

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


class ChatGPTUsageLimitError(ChatGPTPlanError):
    pass


class ChatGPTTemporaryError(Exception):
    """Transient failure (service unavailable, network); safe to retry with backoff."""


_NON_RETRYABLE_HINTS = {
    "subscription_sharing_user_not_eligible": "This ChatGPT account is not eligible for plan usage in apps.",
    "subscription_sharing_unsupported_capability": "The request used a feature not supported with ChatGPT plan usage",
    "subscription_sharing_route_not_supported": "Unsupported endpoint for ChatGPT plan usage.",
    "chatpass_v2_scope_not_authorized": "Plan usage scope not authorized; run login again with --reconsent.",
    "chatpass_v2_invalid_authorization_context": "Invalid authorization context; run login again.",
}
_TEMPORARY_CODES = {
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
}

_notice_lock = threading.Lock()
_notice_shown = False


def _announce_plan_usage():
    global _notice_shown
    with _notice_lock:
        if not _notice_shown:
            print(f"Using ChatGPT plan. Manage usage: {USAGE_SETTINGS_URL}")
            _notice_shown = True


def model_slug(model_name: str) -> str:
    return model_name.split("/", 1)[1] if model_name.startswith(f"{PROVIDER}/") else model_name


def to_responses_input(messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Converts OpenAI chat messages (as built by utils.llm) to Responses API input.

    System messages become `instructions` (explicit system items are rejected).
    """
    instructions = []
    items = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            instructions.append(content if isinstance(content, str) else str(content))
            continue
        if isinstance(content, str):
            parts = [{"type": "input_text", "text": content}]
        else:
            parts = []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    parts.append({"type": "input_text", "text": block["text"]})
                elif kind == "image_url":
                    parts.append({"type": "input_image", "image_url": block["image_url"]["url"]})
                elif kind == "file":
                    f = block["file"]
                    parts.append(
                        {
                            "type": "input_file",
                            "filename": f.get("filename") or "document.pdf",
                            "file_data": f["file_data"],
                        }
                    )
                else:
                    raise ValueError(f"Unsupported content block for ChatGPT plan: {kind}")
        items.append({"role": role, "content": parts})
    return ("\n\n".join(instructions) or None), items


def _error_code(error: Any) -> Optional[str]:
    if error is None:
        return None
    if isinstance(error, dict):
        return error.get("code")
    return getattr(error, "code", None)


def _raise_for_code(code: Optional[str], status: Optional[int], detail: str = ""):
    if code == "subscription_sharing_usage_limit_exceeded":
        raise ChatGPTUsageLimitError(
            "ChatGPT plan usage limit reached (this may be a per-app limit). "
            f"Review your plan or this app's limit: {USAGE_SETTINGS_URL}",
            code,
        )
    if code in _TEMPORARY_CODES or status == 503:
        raise ChatGPTTemporaryError(f"ChatGPT plan usage temporarily unavailable ({code or status}).")
    if code in _NON_RETRYABLE_HINTS:
        hint = _NON_RETRYABLE_HINTS[code]
        if code == "subscription_sharing_unsupported_capability" and detail:
            hint += f" ({detail})"
        raise ChatGPTPlanError(hint, code)
    if code == "subscription_sharing_invalid_user" or status == 401:
        raise chatgpt_auth.ChatGPTLoginRequired(
            "ChatGPT rejected the credentials. Sign in again: uv run python -m utils.chatgpt_auth login"
        )
    if status == 403:
        raise ChatGPTPlanError(f"ChatGPT plan usage blocked ({code or 'forbidden'}). {detail}".strip(), code)
    if status is not None and status >= 500:
        raise ChatGPTTemporaryError(f"ChatGPT service error (HTTP {status}).")
    raise ChatGPTPlanError(f"ChatGPT plan request failed: {code or status} {detail}".strip(), code)


def _client(token: str) -> OpenAI:
    return OpenAI(api_key=token, base_url=chatgpt_auth.API_RESOURCE, max_retries=0)


def _stream_text(client: OpenAI, model: str, instructions: Optional[str], items: List[Dict[str, Any]], timeout) -> str:
    kwargs: Dict[str, Any] = {"model": model, "input": items, "store": False, "stream": True}
    if instructions:
        kwargs["instructions"] = instructions
    if timeout:
        kwargs["timeout"] = timeout
    chunks: List[str] = []
    completed = False
    with client.responses.create(**kwargs) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "response.output_text.delta":
                chunks.append(event.delta)
            elif etype == "response.failed":
                error = getattr(getattr(event, "response", None), "error", None)
                _raise_for_code(_error_code(error) or "unknown_error", None)
            elif etype == "response.incomplete":
                details = getattr(getattr(event, "response", None), "incomplete_details", None)
                reason = getattr(details, "reason", None) or "unknown"
                raise ChatGPTTemporaryError(f"ChatGPT response incomplete ({reason}).")
            elif etype == "response.completed":
                completed = True
    if not completed:
        raise ChatGPTTemporaryError("ChatGPT stream ended before response.completed.")
    return "".join(chunks)


def complete(model_name: str, messages: List[Dict[str, Any]], timeout: Optional[float] = None) -> str:
    """Runs one non-conversational completion on the user's ChatGPT plan."""
    _announce_plan_usage()
    instructions, items = to_responses_input(messages)
    model = model_slug(model_name)

    for attempt in range(2):  # second attempt only after a forced token refresh
        token = chatgpt_auth.get_access_token(force_refresh=attempt > 0)
        try:
            return _stream_text(_client(token), model, instructions, items, timeout)
        except APIStatusError as e:
            body = e.body if isinstance(e.body, dict) else {}
            error = body.get("error") if isinstance(body.get("error"), dict) else None
            code = _error_code(error)
            detail = (error or {}).get("param") or body.get("detail") or ""
            if e.status_code == 401 and attempt == 0 and code in (None, "subscription_sharing_invalid_user"):
                continue  # access token may have been revoked/rotated: refresh once
            _raise_for_code(code, e.status_code, str(detail))
        except APIConnectionError as e:
            raise ChatGPTTemporaryError(f"Network error talking to ChatGPT: {e}") from e
    raise chatgpt_auth.ChatGPTLoginRequired(
        "ChatGPT rejected the credentials after refresh. Sign in again."
    )

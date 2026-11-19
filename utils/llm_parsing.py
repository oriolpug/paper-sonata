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

"""Provider-independent parsers for LLM text responses."""

import json
import re
from typing import Any, Dict, Optional

try:
    from json_repair import repair_json  # type: ignore
except ImportError:  # pragma: no cover - json_repair is a core dependency
    repair_json = None


def identity_parse(response: str):
    return response


def _clean_json_string(json_string: str) -> str:
    json_string = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
    json_string = re.sub(r",\s*([\]}])", r"\1", json_string)
    json_string = re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", json_string)
    return json_string


def _try_load(json_string: str) -> Optional[Any]:
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_clean_json_string(json_string))
    except json.JSONDecodeError:
        return None


def _strip_thinking(response: str) -> str:
    # Some open-source reasoning models (e.g. Qwen3, DeepSeek-R1) inline their
    # chain of thought in <think> tags; it often contains stray braces.
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)


def parse_json(response: str):
    """Extracts the first valid JSON object/array from an LLM response."""
    if not response or not isinstance(response, str):
        return None
    response = _strip_thinking(response)

    matches = re.findall(r"```json(.*?)```", response, re.DOTALL)
    if not matches:
        matches = re.findall(r"(\{.*\}|\[.*\])", response, re.DOTALL)

    for json_string in matches:
        parsed = _try_load(json_string.strip())
        if parsed is not None:
            return parsed

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start_idx = response.find(open_char)
        end_idx = response.rfind(close_char)
        if start_idx != -1 and end_idx > start_idx:
            parsed = _try_load(response[start_idx : end_idx + 1])
            if parsed is not None:
                return parsed

    # Last resort: let json_repair fix common open-model mistakes
    # (unquoted keys, single quotes, truncated output, ...).
    if repair_json is not None:
        candidate = matches[0] if matches else response
        try:
            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, (dict, list)) and repaired:
                return repaired
        except Exception:
            pass

    return None


def parse_latex(response: str):
    response = _strip_thinking(response or "")
    match = re.search(r"```latex\s*(.*?)\s*```", response.strip(), re.DOTALL)
    if match:
        return match.group(1)
    return response


def parse_json_latex(response_text: str) -> str:
    """
    Extracts the JSON worklog and the LaTeX code from the LLM response.
    Expected format:
      ... arbitrary text ...
      ```json
      { ... }
      ```
      ... arbitrary text ...
      ```latex
      ...
      ```
    Returns a JSON string with keys "json" and "latex".
    """
    response_text = _strip_thinking(response_text or "")
    result: Dict[str, Any] = {"json": None, "latex": None}

    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        parsed = _try_load(json_match.group(1))
        if parsed is None:
            print(" >> Error: Failed to parse JSON worklog from response.")
            parsed = {"error": "Invalid JSON format returned"}
        result["json"] = parsed

    latex_match = re.search(r"```latex\s*(.*?)\s*```", response_text, re.DOTALL)
    if latex_match:
        result["latex"] = latex_match.group(1)
    else:
        print(" >> Warning: No ```latex``` block found.")

    return json.dumps(result)

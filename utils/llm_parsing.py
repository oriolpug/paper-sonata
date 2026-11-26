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


def parse_latex_document(response: str):
    """Like parse_latex, but returns None unless the result is a complete
    LaTeX document, so callers retry instead of saving prose or a truncated
    answer (common when a small model loses its instructions)."""
    latex = parse_latex(response)
    if "\\begin{document}" in latex and "\\end{document}" in latex:
        return latex
    return None


_SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
# Where a template section's body ends if no further \section follows.
_SECTION_END_RE = re.compile(
    r"\\section\*?\{|\\appendix\b|\\bibliographystyle\b|\\bibliography\b|"
    r"\\end\{document\}|^\{\s*$",
    re.MULTILINE,
)


def _section_key(title: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"^[\d.\s]+", "", title.lower()))


def splice_sections(template: str, fragment: str):
    """Replaces template sections with same-titled sections from `fragment`.

    Small models often return only the sections they wrote instead of the
    whole document; this merges them into the template. Returns None if the
    fragment contains no section that matches the template.
    """
    blocks = {}
    matches = list(_SECTION_RE.finditer(fragment or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(fragment)
        tail = _SECTION_END_RE.search(fragment, m.end(), end)
        blocks[_section_key(m.group(1))] = fragment[m.start(): tail.start() if tail else end].rstrip() + "\n\n"
    if not blocks:
        return None

    result, replaced, pos = [], 0, 0
    for m in _SECTION_RE.finditer(template):
        if m.start() < pos:
            continue
        key = _section_key(m.group(1))
        if key not in blocks:
            continue
        end = _SECTION_END_RE.search(template, m.end())
        end_pos = end.start() if end else len(template)
        result.append(template[pos:m.start()])
        result.append(blocks[key])
        pos = end_pos
        replaced += 1
    if not replaced:
        return None
    result.append(template[pos:])
    return "".join(result)


_MARKER_RE = re.compile(r"^%{3,}\s*([A-Za-z][A-Za-z \-]*?)\s*%{3,}\s*$", re.MULTILINE)


def splice_marked_blocks(template: str, fragment: str):
    """Fills template placeholders delimited by comment markers such as
    `%%%%%%%%% INTRODUCTION %%%%%%%%%` with the text that follows the same
    marker in `fragment`. Returns None if no marker matches."""
    blocks = {}
    marks = list(_MARKER_RE.finditer(fragment or ""))
    for i, m in enumerate(marks):
        name = m.group(1).strip().upper()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(fragment)
        body = fragment[m.end():end].strip("\n")
        if body.strip() and name not in blocks:
            blocks[name] = body
    if not blocks:
        return None

    result, pos, replaced = [], 0, 0
    marks = list(_MARKER_RE.finditer(template))
    i = 0
    while i < len(marks) - 1:
        opening, closing = marks[i], marks[i + 1]
        name = opening.group(1).strip().upper()
        if name == closing.group(1).strip().upper() and name in blocks:
            result.append(template[pos:opening.end()])
            result.append("\n" + blocks[name] + "\n")
            pos = closing.start()
            replaced += 1
            i += 2
        else:
            i += 1
    if not replaced:
        return None
    result.append(template[pos:])
    return "".join(result)


def make_document_parser(template: str, validate=None):
    """Parser for writing steps: accepts a full LaTeX document, or sections
    that can be spliced into `template`; otherwise returns None (retry).

    `validate(document)` may return a list of problems; a non-empty list
    rejects the answer too (so it is retried).
    """

    def parse(response: str):
        document = parse_latex_document(response)
        if document is None and template:
            fragment = parse_latex(response)
            document = splice_sections(template, fragment) or splice_marked_blocks(template, fragment)
        if document is None:
            return None
        problems = validate(document) if validate else []
        if problems:
            print(f" >> Rejected LaTeX answer: {'; '.join(problems)}")
            return None
        return document

    return parse


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

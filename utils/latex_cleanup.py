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

"""Deterministic fixes for common mistakes in LLM-written LaTeX."""

import re
from typing import Dict, Iterable, List, Tuple

from thefuzz import process  # type: ignore

# Minimum fuzzy similarity (0-100) to map an unknown citation key to a real one.
KEY_MATCH_THRESHOLD = 88

_CITE_RE = re.compile(r"(~?)\\(cite[a-zA-Z]*)(\[[^\]]*\])?(\[[^\]]*\])?\{([^}]*)\}")
_NUMBERED_TITLE_RE = re.compile(
    r"(\\(?:sub)*section\*?\{)\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+"
)


def fix_citation_keys(latex: str, valid_keys: Iterable[str]) -> Tuple[str, List[str]]:
    """Maps unknown \\cite keys to the closest valid key, or drops them.

    Returns the fixed LaTeX and a list of human-readable changes.
    """
    valid = list(dict.fromkeys(valid_keys))
    valid_set = set(valid)
    changes: List[str] = []
    resolved: Dict[str, str] = {}

    def resolve(key: str) -> str:
        if key in valid_set or not valid:
            return key if key in valid_set else ""
        if key not in resolved:
            match = process.extractOne(key, valid)
            if match and match[1] >= KEY_MATCH_THRESHOLD:
                resolved[key] = match[0]
                changes.append(f"citation key '{key}' -> '{match[0]}'")
            else:
                resolved[key] = ""
                changes.append(f"removed unknown citation key '{key}'")
        return resolved[key]

    def replace(m: re.Match) -> str:
        tilde, command, opt1, opt2, keys = m.groups()
        fixed = []
        for key in (k.strip() for k in keys.split(",")):
            new = resolve(key) if key else ""
            if new and new not in fixed:
                fixed.append(new)
        if not fixed:
            return ""
        return f"{tilde}\\{command}{opt1 or ''}{opt2 or ''}{{{', '.join(fixed)}}}"

    return _CITE_RE.sub(replace, latex), changes


def strip_manual_section_numbers(latex: str) -> str:
    """`\\subsection{2.1 Static Models}` -> `\\subsection{Static Models}`:
    LaTeX numbers sections itself, so copied outline numbers appear twice."""
    return _NUMBERED_TITLE_RE.sub(r"\1", latex)


def clean_generated_latex(latex: str, valid_keys: Iterable[str]) -> Tuple[str, List[str]]:
    latex = strip_manual_section_numbers(latex)
    return fix_citation_keys(latex, valid_keys)

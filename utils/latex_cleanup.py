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


_LATEX_SPECIAL = re.compile(r"(?<!\\)([&%#_$])")


def _escape_caption(text: str) -> str:
    return _LATEX_SPECIAL.sub(r"\\\1", text or "")


def ensure_figures_included(
    latex: str, figures: List[Dict], figures_dir: str = "figures"
) -> Tuple[str, List[str]]:
    """Inserts a figure environment for every provided figure the draft never
    includes, before the Conclusion section (or \\end{document}).

    `figures` items have "name" (file name) and optional "caption".
    """
    included = set(re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", latex))
    included_names = {path.rsplit("/", 1)[-1] for path in included}
    blocks, changes = [], []
    for fig in figures or []:
        name = (fig.get("name") or "").strip()
        if not name or name in included_names or name.rsplit(".", 1)[0] in included_names:
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "_", name.rsplit(".", 1)[0]).strip("_")
        blocks.append(
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{{figures_dir}/{name}}}\n"
            f"  \\caption{{{_escape_caption(fig.get('caption', ''))}}}\n"
            f"  \\label{{fig:{stem}}}\n"
            "\\end{figure}\n\n"
        )
        changes.append(f"inserted missing figure '{name}'")
    if not blocks:
        return latex, changes

    anchor = re.search(r"\\section\*?\{\s*(?:\d+\.?\s*)?Conclusions?\s*\}", latex, re.IGNORECASE)
    if not anchor:
        anchor = re.search(r"\\end\{document\}", latex)
    pos = anchor.start() if anchor else len(latex)
    return latex[:pos] + "".join(blocks) + latex[pos:], changes


# Table cells too generic to signal copied example content.
_GENERIC_CELLS = {"method", "methods", "ours", "model", "models", "setting", "results", "part", "description"}


def example_phrases(guidelines: str, template: str = "") -> Tuple[List[str], List[str]]:
    """Content that must never appear in a finished paper.

    Returns (example_cells, placeholders): text of the example tables in the
    conference guidelines (e.g. "Frobnability"), and the template's
    placeholder lines (e.g. "Methodology here.").
    """
    cells = set()
    for block in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", guidelines or "", re.S):
        body = re.sub(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}|\\end\{tabular\}", " ", block)
        body = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}", " ", body)
        for row in re.split(r"\\\\", body):
            for cell in row.split("&"):
                text = re.sub(r"\\(?:bf|it|em|textbf|textit|emph|hline|toprule|midrule|bottomrule)\b|[{}]", " ", cell)
                text = " ".join(text.split())
                if (
                    len(text) >= 4
                    and not re.fullmatch(r"[\d.\s%±+\-]+", text)
                    and text.lower() not in _GENERIC_CELLS
                ):
                    cells.add(text)
    placeholders = sorted(
        {line.strip() for line in (template or "").splitlines() if re.fullmatch(r"[A-Z][\w ,]* here\.", line.strip())}
    )
    return sorted(cells), placeholders


def find_copied_examples(latex: str, guidelines: str, template: str = "") -> List[str]:
    """Problems if the draft copied guideline example tables (two or more
    example cells) or left template placeholders in place."""
    cells, placeholders = example_phrases(guidelines, template)
    problems = []
    copied = [c for c in cells if c in latex]
    if len(copied) >= 2:
        problems.append(f"copied example table content from the guidelines: {copied[:4]}")
    left = [p for p in placeholders if p in latex]
    if left:
        problems.append(f"template placeholders left: {left}")
    return problems

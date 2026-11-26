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

"""Shrinks paper lists in prompts to fit small context windows (local models).

Large hosted models get the full prompt unchanged. For small windows, paper
abstracts are shortened step by step, then the lowest-priority papers are
dropped, until the rendered prompt fits the token budget.
"""

from typing import Callable, Optional, Tuple

from utils.llm import count_tokens, prompt_budget

# Abstract length (characters) tried in order; None = unchanged.
ABSTRACT_LEVELS = (None, 600, 250, 0)


def truncate_abstract(abstract: str, max_chars: Optional[int]) -> str:
    abstract = abstract or ""
    if max_chars is None or len(abstract) <= max_chars:
        return abstract
    if max_chars == 0:
        return ""
    return abstract[:max_chars].rsplit(" ", 1)[0] + "..."


def fit_papers(
    model_name: str,
    render: Callable[[Optional[int], int], str],
    n_papers: int,
    output_reserve: int,
) -> Tuple[str, Optional[int], int]:
    """Renders a prompt that fits the model's context.

    `render(abstract_chars, n)` must build the full prompt text using the
    first `n` papers (callers order papers by priority) with abstracts cut to
    `abstract_chars`. Returns (text, abstract_chars, n).
    """
    budget = prompt_budget(model_name, output_reserve)
    if budget is None:
        return render(None, n_papers), None, n_papers

    for level in ABSTRACT_LEVELS:
        text = render(level, n_papers)
        if count_tokens(model_name, text) <= budget:
            return text, level, n_papers

    n = n_papers
    while n > 1:
        n = max(1, int(n * 0.8))
        text = render(0, n)
        if count_tokens(model_name, text) <= budget:
            return text, 0, n
    return render(0, n), 0, n


def describe_fit(n_total: int, abstract_chars: Optional[int], n_used: int) -> Optional[str]:
    if abstract_chars is None and n_used == n_total:
        return None
    parts = []
    if abstract_chars is not None:
        parts.append(f"abstracts cut to {abstract_chars} chars" if abstract_chars else "abstracts omitted")
    if n_used < n_total:
        parts.append(f"{n_used}/{n_total} papers kept")
    return "Prompt shrunk to fit the model's context: " + ", ".join(parts)

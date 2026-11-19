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

"""Default model names, overridable via environment variables (or `.env`).

Model names follow LiteLLM conventions: `<provider>/<model>`, e.g.
`gemini/gemini-3.1-pro-preview`, `openai/gpt-5`, `anthropic/claude-sonnet-5-5`,
`ollama/qwen3:32b`, `hosted_vllm/meta-llama/Llama-3.3-70B-Instruct`.
Bare legacy names (e.g. `gemini-3.1-pro-preview`) are still accepted.
"""

import os

from dotenv import load_dotenv  # type: ignore

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Main model for writing, outlining and literature synthesis.
DEFAULT_WRITER_MODEL = os.getenv("PAPER_WRITER_MODEL", "gemini-3.1-pro-preview")
# Model for reviewing / refinement loops.
DEFAULT_REFLECTION_MODEL = os.getenv("PAPER_REFLECTION_MODEL", DEFAULT_WRITER_MODEL)
# Model for planning and writing plot code.
DEFAULT_PLOTTING_MODEL = os.getenv("PAPER_PLOTTING_MODEL", DEFAULT_WRITER_MODEL)
# Cheaper model for high-volume tasks (literature discovery, parsing).
DEFAULT_FAST_MODEL = os.getenv("PAPER_FAST_MODEL", "gemini-3-flash-preview")
# Image generation model for conceptual diagrams. Set to "none" to skip diagrams.
DEFAULT_IMAGE_MODEL = os.getenv("PAPER_IMAGE_MODEL", "gemini-3-pro-image-preview")
# Model used by the autoraters (evaluation scripts).
DEFAULT_EVAL_MODEL = os.getenv("PAPER_EVAL_MODEL", "gemini-3.1-pro-preview")


def _env_set(name: str) -> bool:
    return bool(os.getenv(name))


def resolve_model_names(
    writer: str = None,
    reflection: str = None,
    plotting: str = None,
    fast: str = None,
    image: str = None,
) -> dict:
    """Fills in unspecified model names.

    Explicit arguments win, then PAPER_*_MODEL environment variables. Otherwise
    derived models follow the writer model, so that choosing e.g. an Ollama
    writer never silently sends traffic to Gemini. The Gemini fast/image
    defaults are only used when the writer is itself a Gemini model.
    """
    from utils.llm import is_gemini_model

    writer = writer or DEFAULT_WRITER_MODEL
    writer_is_gemini = is_gemini_model(writer)

    def pick(value, env_name, env_default, fallback):
        if value:
            return value
        if _env_set(env_name):
            return env_default
        return fallback

    return {
        "writer": writer,
        "reflection": pick(reflection, "PAPER_REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL, writer),
        "plotting": pick(plotting, "PAPER_PLOTTING_MODEL", DEFAULT_PLOTTING_MODEL, writer),
        "fast": pick(
            fast,
            "PAPER_FAST_MODEL",
            DEFAULT_FAST_MODEL,
            DEFAULT_FAST_MODEL if writer_is_gemini else writer,
        ),
        "image": pick(
            image,
            "PAPER_IMAGE_MODEL",
            DEFAULT_IMAGE_MODEL,
            DEFAULT_IMAGE_MODEL if writer_is_gemini else "none",
        ),
    }

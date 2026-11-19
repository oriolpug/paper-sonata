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

PAPER_ID="cvpr2025_0b62029e18" # Switch this to the paper id you want to write

# Python launcher. Defaults to `uv run python`; override e.g. with
#   PYTHON=.venv/bin/python ./paper_writing_cli.sh
PYTHON="${PYTHON:-uv run python}"

# Pick any LiteLLM model, e.g. gemini/gemini-3.1-pro-preview, openai/gpt-5,
# anthropic/claude-sonnet-5-5, ollama_chat/qwen3:32b
WRITER_MODEL="${WRITER_MODEL:-gemini-3.1-pro-preview}"

PYTHONUNBUFFERED=1 nohup $PYTHON paper_writing_cli.py \
--writer_model_name "${WRITER_MODEL}" \
--raw_materials_dir datasets/cvpr2025/papers/${PAPER_ID}/raw_materials \
--latex_template_dir templates/cvpr2025 \
--output_dir ./paper_output_${PAPER_ID} \
--use_plotting true \
> ./paper_writing_${PAPER_ID}.log 2>&1 &



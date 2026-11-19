# <div align="center">🎻 PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing</div>
<div align="center">Yiwen Song<sup>1</sup>, Yale Song<sup>1</sup>, Tomas Pfister<sup>1</sup>, and Jinsung Yoon<sup>1</sup></div>
<div align="center"><sup>1</sup>Google Cloud AI Research</div>
<br><br>

PaperOrchestra is a multi-agent framework for automated AI research paper writing. It flexibly transforms unconstrained pre-writing materials (ideas and experimental logs) into submission-ready LaTeX manuscripts, including comprehensive literature synthesis and generated visuals, such as plots and conceptual diagrams.

Acting like an orchestrated team of specialized agents, it handles outline generation, literature review, section writing, content refinement, and plotting to close the gap between raw research materials and submission ready papers.

<div align="center">
  <img src="assets/overview.png" alt="PaperOrchestra Overview" width="90%"/>
</div>

## Key Features

- **Multi-Agent Pipeline**: Orchestrates specialized agents (Outline, Literature Review, Section Writing, Content Refinement, Plotting) for end-to-end paper generation.
- **Comprehensive Literature Synthesis**: Automatically searches, prioritizes, and integrates relevant literature into the manuscript.
- **Visuals Generation**: Generates plots and supports the inclusion of conceptual diagrams.
- **LaTeX Automation**: Produces submission-ready LaTeX manuscripts adhering to target conference templates.

## Installation

No conda required. A LaTeX distribution providing `pdflatex`/`bibtex` (e.g. TeX Live or MacTeX) must be on your `PATH` to compile papers.

1.  **Environment Setup** (Python >= 3.11). Recommended, with [uv](https://docs.astral.sh/uv/):
    ```bash
    uv sync                    # creates .venv and installs dependencies
    uv sync --extra frontend   # also installs the Streamlit demo
    ```
    Or with plain `venv` + `pip`:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Environment Variables**: Copy `.env.example` to `.env` in the project root (`cp .env.example .env`; it is gitignored) and set the keys for the providers you use:
    ```env
    # Literature search (all optional). Papers are looked up in Semantic Scholar,
    # falling back to OpenAlex and then arXiv when a source is rate-limited or has
    # no match. Keyless access works but is shared and often rate-limited (HTTP 429);
    # free keys make runs faster and more complete.
    SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
    OPENALEX_API_KEY=your_openalex_api_key    # keyless: ~100 keyword searches/day per IP
    # PAPER_SEARCH_SOURCES=s2,openalex,arxiv  # order/subset; drop arxiv to exclude preprints-only hits

    # Hosted providers: set only the ones you use
    GEMINI_API_KEY=your_gemini_api_key
    # ...or Gemini via Vertex AI (takes precedence for bare "gemini-*" names)
    VERTEX_AI_PROJECT=your_vertex_ai_project
    VERTEX_AI_LOCATION=your_vertex_ai_location
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key

    # Self-hosted / open-source models
    OLLAMA_API_BASE=http://localhost:11434
    HOSTED_VLLM_API_BASE=http://localhost:8000/v1
    LM_STUDIO_API_BASE=http://localhost:1234/v1

    # Optional: default models (otherwise pass them on the command line)
    # PAPER_WRITER_MODEL=ollama_chat/qwen3:32b
    # PAPER_FAST_MODEL=ollama_chat/qwen3:8b
    # PAPER_IMAGE_MODEL=none

    # if you want to run demo and send emails to yourself at the end, set these 2 fields as well
    SMTP_EMAIL=your_email
    SMTP_PASSWORD=your_password
    ```

## Choosing a Model

All LLM calls go through [LiteLLM](https://docs.litellm.ai/docs/providers), so any provider it supports works. Model names use LiteLLM's `<provider>/<model>` format:

| Provider | Example model name |
|---|---|
| Gemini API / Vertex AI | `gemini/gemini-3.1-pro-preview`, `vertex_ai/gemini-3.1-pro-preview` |
| OpenAI | `openai/gpt-5` |
| Anthropic | `anthropic/claude-sonnet-5-5` |
| OpenRouter | `openrouter/<org>/<model>` |
| Ollama | `ollama_chat/qwen3:32b` |
| vLLM | `hosted_vllm/Qwen/Qwen3-32B` |
| LM Studio | `lm_studio/<model>` |
| Any OpenAI-compatible server | `openai/<model>` with `OPENAI_API_BASE` set |
| Your ChatGPT plan (Plus/Pro), no API key | `chatgpt/<model>` (see below) |

Bare legacy names such as `gemini-3.1-pro-preview`, `gpt-5` or `claude-...` still work.

Behaviour that adapts to the model:
*   **PDF inputs** (refinement, autoraters) are sent natively to models that accept PDFs, and as extracted text otherwise.
*   **Images** are dropped (with a warning) for models known not to support vision. For local models LiteLLM doesn't know about, images are sent; set `PAPER_LLM_SUPPORTS_VISION=true/false` to force either behaviour.
*   **Literature discovery** uses Gemini's Google Search grounding when the fast model is Gemini; otherwise the model writes search queries and selects papers from the real search results (`--lit_search_backend`).
*   **Paper verification** (every cited paper, in all modes) checks Semantic Scholar, then OpenAlex, then arXiv, and runs in the background while discovery is still going. When a paper exists in several versions, the published one is preferred over the preprint; papers found only on arXiv are cited as `arXiv preprint arXiv:<id>`. A source that is rate-limited is skipped for a while (`PAPER_SEARCH_COOLDOWN`, default 300s, or the server's requested wait).
*   **Citation recovery**: when a suggested paper can't be found by title, the fast model proposes corrected titles/queries, the databases are searched again, and the model picks the real paper or rejects the candidate as non-existent. Limited to `--max_citation_recoveries` attempts per run (default 40, `0` disables). Every recovered and rejected candidate, with the reason, is listed in `literature_agent_output/candidate_verification.json`.
*   **No API keys are required.** Keyless Semantic Scholar shares one global pool and is often fully rate-limited; OpenAlex title checks (autocomplete + lookup by ID) don't use its daily budget, only keyword searches do; arXiv is free but slow (1 request / 3 s).
*   **Diagrams** need an image-generation model (`--image_model_name`, e.g. `gemini/gemini-3-pro-image-preview` or `openai/gpt-image-1`). With `none`, only code-generated plots are produced.

### Using your ChatGPT plan instead of an API key

ChatGPT Plus/Pro subscribers can run the pipeline on their plan through OpenAI's official [Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source) for open-source apps (currently a preview feature):

```bash
uv run python -m utils.chatgpt_auth login    # "Continue with ChatGPT" in your browser (one time)
uv run python -m utils.chatgpt_auth models   # models available to your plan
uv run python paper_writing_cli.py ... --writer_model_name chatgpt/<model-slug>
```

*   Usage counts against your ChatGPT plan: on Plus the 5-hour limit is shared with ChatGPT and your other apps; Pro has no 5-hour limit. Review usage or set a limit for this app at https://chatgpt.com/settings/usage. When the limit is hit, the run stops with a message instead of switching to paid API usage.
*   Image generation is not available on plan usage: diagrams need an API image model (`--image_model_name openai/gpt-image-1` or a Gemini image model) or `none`. `temperature` and output-length settings are ignored.
*   Credentials are stored with owner-only permissions in `~/.config/paper-orchestra/` (override with `PAPER_CHATGPT_CONFIG_DIR`) and refreshed automatically. `uv run python -m utils.chatgpt_auth logout` revokes and deletes them.
*   This uses OpenAI's documented flow, not LiteLLM's built-in `chatgpt/` provider (which reuses Codex CLI credentials). Claude Pro/Max subscriptions can't be used this way; Anthropic only permits API keys for third-party tools.

Notes for local/open-source models:
*   The pipeline sends long prompts (full template, idea, experimental log, draft paper). Use models with at least 32k context. For Ollama, the context window is set via `PAPER_OLLAMA_NUM_CTX` (default `32768`).
*   Some providers cap output length; set `PAPER_MAX_OUTPUT_TOKENS` if full-paper LaTeX responses get truncated. `PAPER_LLM_TIMEOUT` (seconds) helps with slow local inference.
*   `<think>...</think>` reasoning blocks from models like Qwen3 or DeepSeek-R1 are stripped automatically.

## Dataset

The dataset used for paper generation is **not included** in this code repository. It will be released separately at a later date.
Once available, you should download the dataset and place it under the `datasets/` directory.

## Usage

You can run the paper generation pipeline using the provided bash script or directly via the Python CLI.

### Using the Bash Script
```bash
./paper_writing_cli.sh
# with another model / python launcher:
WRITER_MODEL=ollama_chat/qwen3:32b PYTHON=.venv/bin/python ./paper_writing_cli.sh
```

### Using the Python CLI directly
```bash
uv run python paper_writing_cli.py \
  --raw_materials_dir /path_to_your_raw_materials/ \
  --latex_template_dir /path_to_your_latex_template/ \
  --output_dir /path_to_your_output_folder/ \
  --use_plotting true
```

Example with a local open-source model via Ollama:
```bash
uv run python paper_writing_cli.py \
  --raw_materials_dir /path_to_your_raw_materials/ \
  --latex_template_dir templates/cvpr2025 \
  --writer_model_name ollama_chat/qwen3:32b \
  --use_plotting true --image_model_name none
```

### Key Arguments
*   `--raw_materials_dir`: (Required) Directory containing the raw materials (e.g., idea and experimental log).
*   `--latex_template_dir`: (Required) Directory containing the LaTeX template (e.g., `templates/cvpr2025`).
*   `--output_dir`: Directory to output the generated paper. If not specified, an automatic folder with a timestamp is created.
*   `--use_plotting`: Enable the plotting agent workflow to generate figures from code (Default: `False`).
*   `--writer_model_name`: LLM for writer and literature agents (any LiteLLM model name, see above).
*   `--reflection_model_name`: LLM for reflection agents (default: same as writer).
*   `--plotting_model_name`: LLM for the plotting agent (default: same as writer).
*   `--fast_model_name`: Cheaper LLM for literature discovery (default: `gemini-3-flash-preview` for Gemini writers, otherwise same as writer).
*   `--lit_search_backend`: `auto` (default), `google` (Gemini only) or `s2` (Semantic Scholar, any model).
*   `--image_model_name`: Image generation model for diagrams, or `none` (default: Gemini image model for Gemini writers, otherwise `none`).

### Interactive Demo

We provide an interactive Streamlit demo for the paper generation pipeline. For instructions on how to run and use it, please refer to the [README.md](frontend/README.md) in the `frontend` directory.

## Project Structure

*   `autoraters/`: Contains scripts for automated evaluation.
*   `methods/`: Core logic for agents and pipeline execution.
*   `templates/`: LaTeX templates for target conferences, you can add your own template by creating a new subfolder under this directory.
*   `utils/`: Utility functions for API calls, PDF handling, etc.

## Citation

If you find this repo or our paper helpful, please cite it as follows:

```bibtex
@article{song2026paperorchestra,
  title={PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing},
  author={Song, Yiwen and Song, Yale and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2604.05018},
  year={2026}
}
```

# Disclaimer

This is not an officially supported Google product.

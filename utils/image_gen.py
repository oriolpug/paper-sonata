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

"""Provider-agnostic image generation built on LiteLLM."""

import base64
import os
import time
from typing import Optional

import litellm  # type: ignore
import requests

from utils.llm import (
    _completion_kwargs,
    get_provider,
    is_disabled_model,
    normalize_model_name,
)

# OpenAI-style sizes closest to common aspect ratios.
_OPENAI_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "4:3": "1536x1024",
    "3:2": "1536x1024",
    "9:16": "1024x1536",
    "3:4": "1024x1536",
    "2:3": "1024x1536",
}


def _image_params(model: str, aspect_ratio: str) -> dict:
    provider = get_provider(model)
    if provider in ("gemini", "vertex_ai"):
        return {"imageConfig": {"aspectRatio": aspect_ratio}}
    return {"size": _OPENAI_SIZES.get(aspect_ratio, "1024x1024")}


def _first_image_b64(response) -> Optional[str]:
    for item in getattr(response, "data", None) or []:
        b64 = getattr(item, "b64_json", None)
        if b64:
            return b64
        url = getattr(item, "url", None)
        if url:
            if url.startswith("data:"):
                return url.split(",", 1)[1]
            rsp = requests.get(url, timeout=60)
            rsp.raise_for_status()
            return base64.b64encode(rsp.content).decode("utf-8")
    return None


def generate_image(
    model_name: Optional[str],
    prompt: str,
    aspect_ratio: str = "16:9",
    system_prompt: Optional[str] = None,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Generates an image and returns it base64-encoded, or None on failure.

    Pass model_name="none" (or None) to disable image generation.
    """
    if is_disabled_model(model_name):
        print("Image generation disabled (image model is 'none'); skipping diagram.")
        return None

    model = normalize_model_name(model_name)
    if model.startswith("chatgpt/"):
        print(
            "Image generation is not available with ChatGPT plan usage; skipping diagram. "
            "Use an API image model (e.g. openai/gpt-image-1) or 'none'."
        )
        return None
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    params = _image_params(model, aspect_ratio)
    params.update(
        {
            k: v
            for k, v in _completion_kwargs(model).items()
            if k in ("vertex_project", "vertex_location", "timeout")
        }
    )

    for attempt in range(max_retries):
        try:
            response = litellm.image_generation(prompt=full_prompt, model=model, **params)
            img_base64 = _first_image_b64(response)
            if img_base64 is None:
                raise ValueError("No image returned by the model")

            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    with open(save_path, "wb") as fh:
                        fh.write(base64.b64decode(img_base64))
                    print(f"Image saved to {save_path}")
                except Exception as e:
                    print(f"Warning: Failed to save image to {save_path}: {e}")
            return img_base64

        except (litellm.AuthenticationError, litellm.NotFoundError) as e:
            print(f"Error: image generation with '{model}' failed: {e}")
            return None
        except Exception as e:
            print(f"Warning: image generation with '{model}' failed: {e}")
            if attempt + 1 == max_retries:
                print(f"Error: All {max_retries} retry attempts failed for image generation.")
                break
            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    return None

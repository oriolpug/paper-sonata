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

"""Backwards-compatible convenience wrappers around `utils.llm.call_llm`.

`generation_configs` accepts the keys used throughout the codebase:
`system_instruction`, `temperature` and `tools`; any other keys are passed
straight to LiteLLM.
"""

from typing import Any, Callable, Dict, List, Optional

from utils import llm
from utils.llm_parsing import identity_parse, parse_json


def get_llm_parser(model_name: str, return_json: bool = True) -> Callable:
    return parse_json if return_json else identity_parse


def call_llm_with_contents(
    contents: List[llm.Content],
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
    max_retries: int = 5,
    base_interval_sec: int = 5,
) -> Dict[str, Any]:
    configs = dict(generation_configs or {})
    system_prompt = configs.pop("system_instruction", None)
    temperature = configs.pop("temperature", None)
    tools = configs.pop("tools", None)
    parser = result_parsing_func or get_llm_parser(model_name, return_json)
    return llm.call_llm(
        contents=contents,
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        tools=tools,
        result_parsing_func=parser,
        check_parsed_response_not_none=check_parsed_response_not_none,
        max_retries=max_retries,
        base_interval_sec=base_interval_sec,
        **configs,
    )


def call_llm_with_text_prompt(
    prompt: str,
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    return call_llm_with_contents(
        contents=[llm.text(prompt)],
        model_name=model_name,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        return_json=return_json,
        result_parsing_func=result_parsing_func,
    )


def call_llm_with_images(
    prompt: str,
    images: list[str],
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    contents: List[llm.Content] = [llm.text(prompt)]
    contents.extend(llm.image(path) for path in images)
    return call_llm_with_contents(
        contents=contents,
        model_name=model_name,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        return_json=return_json,
        result_parsing_func=result_parsing_func,
    )


def call_llm_with_pdf(
    pdf_path: str,
    prompt: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Sends the PDF natively when the model supports it, else its extracted text."""
    generation_configs: Dict[str, Any] = {"temperature": temperature}
    if system_instruction is not None:
        generation_configs["system_instruction"] = system_instruction
    return call_llm_with_contents(
        contents=[llm.pdf(pdf_path), llm.text(prompt)],
        model_name=model_name,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        return_json=return_json,
        result_parsing_func=result_parsing_func,
    )

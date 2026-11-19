import json

from utils.llm_parsing import parse_json, parse_json_latex, parse_latex


def test_parse_json_fenced():
    assert parse_json('x ```json\n{"a": [1, 2]}\n``` y') == {"a": [1, 2]}


def test_parse_json_trailing_comma():
    assert parse_json('{"a": 1,}') == {"a": 1}


def test_parse_json_repair_fallback():
    # single quotes / unquoted keys: common with small open models
    assert parse_json("{'a': 1, b: 'two'}") == {"a": 1, "b": "two"}


def test_parse_json_none():
    assert parse_json("") is None
    assert parse_json(None) is None


def test_parse_latex():
    assert parse_latex("intro\n```latex\n\\section{A}\n```") == "\\section{A}"
    assert parse_latex("\\section{B}") == "\\section{B}"


def test_parse_json_latex():
    out = json.loads(parse_json_latex('```json\n{"log": 1}\n```\n```latex\n\\x\n```'))
    assert out == {"json": {"log": 1}, "latex": "\\x"}

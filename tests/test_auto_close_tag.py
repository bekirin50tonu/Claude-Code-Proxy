"""Unit tests for AutoCloseTagParser (Asynchronous Tag & JSON Auto-Closer)."""

import json

from atomic.parsers.auto_close_tag import AutoCloseTagParser


def test_auto_close_unclosed_quotes() -> None:
    # 1. Unclosed quote at EOF
    truncated_str = '{"file_path": "/media/bekir/HDDStorage/PROJECTS/MY_SITE'
    repaired = AutoCloseTagParser.auto_close_quotes(truncated_str)
    assert repaired == '{"file_path": "/media/bekir/HDDStorage/PROJECTS/MY_SITE"'

    # 2. Closed quote (even count) should be unchanged
    closed_str = '{"key": "value"}'
    assert AutoCloseTagParser.auto_close_quotes(closed_str) == closed_str


def test_auto_close_xml_hermes_tags() -> None:
    # 1. Truncated parameter tag
    truncated_param = "<<parameter=file_path>\n/media/bekir/HDDStorage/PROJECTS/MY_SITE"
    repaired_param = AutoCloseTagParser.auto_close_xml_tags(truncated_param)
    assert "</parameter>" in repaired_param

    # 2. Truncated parameter + function + tool_call hierarchical sequence
    truncated_seq = "<tool_call>\n<<function=write_file>>\n<<parameter=content>>hello world"
    repaired_seq = AutoCloseTagParser.auto_close_xml_tags(truncated_seq)
    assert repaired_seq.endswith("</parameter></function></tool_call>")

    # 3. Truncated <think> tag
    truncated_think = "<think>Analyzing the user codebase for potential bugs..."
    repaired_think = AutoCloseTagParser.auto_close_xml_tags(truncated_think)
    assert repaired_think.endswith("</think>")


def test_auto_close_json_brackets() -> None:
    # 1. Unclosed object braces
    unclosed_obj = '{"summary": "Work in progress", "details": {"status": "ok"'
    repaired_obj = AutoCloseTagParser.auto_close_json_brackets(unclosed_obj)
    assert repaired_obj.endswith("}}")
    parsed = json.loads(repaired_obj)
    assert parsed["details"]["status"] == "ok"

    # 2. Unclosed array and object combination (LIFO stack)
    unclosed_mix = '{"items": [{"name": "item1"}, {"name": "item2"'
    repaired_mix = AutoCloseTagParser.auto_close_json_brackets(unclosed_mix)
    assert repaired_mix.endswith("}]}")
    parsed_mix = json.loads(repaired_mix)
    assert len(parsed_mix["items"]) == 2


def test_full_repair_truncated_stream_pipeline() -> None:
    # Complete truncated stream example with unclosed quotes, parameter/function tags, and brackets
    raw_cutoff = (
        "<tool_call>\n"
        "<<function=replace_file>>\n"
        "<<parameter=TargetFile>>\n"
        '{"path": "/media/bekir/HDDStorage/PROJECTS/MY_SITE/server.py'
    )

    repaired = AutoCloseTagParser.repair_truncated_stream(raw_cutoff)
    assert repaired.endswith('"}</parameter></function></tool_call>')

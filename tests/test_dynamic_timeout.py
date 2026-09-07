"""Unit tests for Dynamic Timeout Calculator."""

from shared.utils.timeout_calculator import calculate_dynamic_timeout


def test_calculate_dynamic_timeout_simple_request() -> None:
    """Simple requests without tools or thinking should receive standard base timeout (min 45s)."""
    t = calculate_dynamic_timeout(model="nvidia_nim/meta/llama-3.1-8b-instruct", max_tokens=1000)
    assert t >= 45.0


def test_calculate_dynamic_timeout_heavy_workload() -> None:
    """Heavy agentic tool requests with thinking models should dynamically scale up timeout."""
    messages = [
        {"role": "user", "content": "wget https://example.com/pages and edit router layout " + "x" * 20000}
    ]
    tools = [{"name": "write"}, {"name": "edit"}, {"name": "bash"}]

    t = calculate_dynamic_timeout(
        model="nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        messages=messages,
        tools=tools,
        max_tokens=16384,
    )

    # Base (600s) + Thinking (+180s) + Tools (+360s) -> should scale up significantly
    assert t >= 600.0
    assert t <= 1800.0

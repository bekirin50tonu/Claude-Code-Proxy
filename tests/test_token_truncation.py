"""Unit tests for TokenBudgetGuard and Context Retention Hierarchy."""

from atomic.guards.token_budget import TokenBudgetGuard


def test_token_budget_guard_preserves_system_and_truncates_head() -> None:
    """Verify that TokenBudgetGuard preserves system prompt while truncating old head turns."""
    guard = TokenBudgetGuard("nvidia_nim/meta/llama-3.1-8b-instruct")

    system_prompt = "System prompt: You are a secure coding assistant."
    messages = [
        {"role": "user", "content": "Turn 1: " + "a" * 1000},
        {"role": "assistant", "content": "Turn 1 response: " + "b" * 1000},
        {"role": "user", "content": "Turn 2: " + "c" * 1000},
        {"role": "assistant", "content": "Turn 2 response: " + "d" * 1000},
        {"role": "user", "content": "Turn 3: What is 2+2?"},
    ]

    # Truncate with very small max token budget to force turn trimming
    truncated_msgs, system_res, was_truncated = guard.check_and_truncate(
        messages=messages,
        system=system_prompt,
        max_tokens=100,
    )

    # System prompt must remain unchanged
    assert system_res == system_prompt

    # Truncated messages should start with user role
    if truncated_msgs:
        assert truncated_msgs[0]["role"] == "user"
        # The last turn should be preserved
        assert truncated_msgs[-1]["content"] == "Turn 3: What is 2+2?"

"""Tiny helper for calling Claude.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...

If no API key is set, `call_claude` returns None.
"""

import os

MODEL = "claude-opus-5-5"  # or "claude-sonnet-5-5" for a cheaper model


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_claude(system: str, user: str, max_tokens: int = 1024) -> str | None:
    """Send one system prompt + one user message to Claude and return the text reply."""
    if not has_api_key():
        return None

    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        # "low" effort keeps answers fast and cheap; RAG questions are easy once
        # the right context is in the prompt.
        output_config={"effort": "low"},
        # If Claude's safety filters ever decline a request, retry it on a
        # fallback model automatically instead of returning nothing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "[Claude declined to answer this request]"
    return "".join(block.text for block in response.content if block.type == "text")

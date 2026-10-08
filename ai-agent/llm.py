"""Helpers for talking to "the model". You don't need to edit this file.

An agent only needs ONE thing from the model: given the conversation so far
and a list of tools, return the next assistant message. Both functions below
do exactly that and return the same shape:

    {"stop_reason": "tool_use" | "end_turn" | ...,
     "content": [{"type": "text", "text": "..."},
                 {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}}]}

  call_model  asks Claude (needs ANTHROPIC_API_KEY; returns None without one).

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import os
import uuid

MODEL = "claude-opus-5-5"  # or "claude-sonnet-5-5" for a cheaper model


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_model(system: str, messages: list[dict], tools: list[dict], max_tokens: int = 16000) -> dict | None:
    """Send the whole conversation plus the tool list to Claude, return its next message."""
    if not has_api_key():
        return None

    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=tools,
        # "low" effort keeps the exercise fast and cheap. Try "medium" or
        # "high" on harder tasks and watch how the tool calls change.
        output_config={"effort": "low"},
        # If Claude's safety filters ever decline a request, retry it on a
        # fallback model automatically instead of returning nothing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return {"stop_reason": "refusal", "content": [{"type": "text", "text": "[Claude declined this request]"}]}
    # Turn the SDK objects into plain dicts. Claude may also return "thinking"
    # blocks; they are kept as-is, because the API wants the assistant's
    # message sent back exactly as it came.
    return {
        "stop_reason": response.stop_reason,
        "content": [block.model_dump(exclude_none=True) for block in response.content],
    }
"""
Shared attacker-LLM plumbing: client construction and a single forced
tool-call helper, used by both attacker/attacker.py's propose_candidates and
attacker/mutation.py's generate_mutations, so the Anthropic/Groq branching
and Groq's tool-call retry logic exist in exactly one place rather than
being copy-pasted into both.

Groq is the project default (see victim/agent.py) - it needs no Anthropic
budget to run. Anthropic remains fully supported; pass provider="anthropic"
explicitly (e.g. once Anthropic credits are available) to use it instead.
"""

import json
import os

import anthropic
from dotenv import load_dotenv
from openai import BadRequestError

from victim.agent import build_groq_client, to_openai_tools

load_dotenv()

DEFAULT_ATTACKER_MODEL_ANTHROPIC = os.environ.get("ATTACKER_MODEL_ANTHROPIC", "claude-sonnet-5")
DEFAULT_ATTACKER_MODEL_GROQ = os.environ.get("ATTACKER_MODEL_GROQ", "llama-3.3-70b-versatile")

# Llama via Groq fails to produce a valid tool call (tool_use_failed, or a
# tool call whose arguments aren't valid JSON) often enough in practice - we
# measured roughly 1 in 8 on the victim side - that a small retry here is
# worth the complexity. This is a helper call feeding the search, not itself
# a scored trial, so retrying doesn't distort any research result the way
# retrying a victim-agent call would.
GROQ_TOOL_CALL_RETRIES = 3


def default_attacker_model(provider: str) -> str:
    return DEFAULT_ATTACKER_MODEL_ANTHROPIC if provider == "anthropic" else DEFAULT_ATTACKER_MODEL_GROQ


def build_attacker_client(provider: str):
    if provider == "anthropic":
        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    if provider == "groq":
        return build_groq_client()  # reads GROQ_API_KEY from env - see victim/agent.py
    raise ValueError(f"Unknown attacker provider '{provider}' - expected 'anthropic' or 'groq'.")


def call_with_forced_tool(client, provider: str, model: str, system_prompt: str, user_prompt: str, tool: dict) -> dict:
    """Call the attacker-LLM forced through `tool` (a single Anthropic-shaped
    tool definition, converted for Groq as needed), and return the parsed
    input/arguments dict. Retries on Groq's tool_use_failed / malformed
    tool-call JSON; Anthropic doesn't show that failure mode here, so its
    path is a single plain call."""
    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user_prompt}],
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        return tool_use.input

    # groq
    attempts = 0
    while True:
        attempts += 1
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=to_openai_tools([tool]),
                tool_choice={"type": "function", "function": {"name": tool["name"]}},
            )
            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)
        except (BadRequestError, json.JSONDecodeError) as exc:
            if attempts >= GROQ_TOOL_CALL_RETRIES:
                raise RuntimeError(
                    f"Attacker-LLM ({model} via Groq) failed to produce a valid '{tool['name']}' "
                    f"tool call after {attempts} attempts: {exc}"
                ) from exc

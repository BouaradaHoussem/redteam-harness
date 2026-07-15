"""
Fake tool implementations for the victim agent.

Nothing here touches a real inbox or calendar. Each tool reads from a small
JSON file in fake_data/ and returns plain text, the same shape a real
integration would hand back to the model. That's the point: from the model's
perspective a fake email and a real one look identical, which is what makes
prompt injection via tool output possible in the first place.

Every call also gets recorded in `self.log` as {name, input, result} — the
agent appends to this on every turn, and scoring/tripwires.py reads it
afterward to check what the agent actually did.
"""

import json
from pathlib import Path

FAKE_DATA_DIR = Path(__file__).parent / "fake_data"

# JSON schemas handed to the Anthropic API. Keep these in sync with the
# ToolBox methods below - the "name" here must match a method name.
TOOL_DEFINITIONS = [
    {
        "name": "read_email",
        "description": "Read the user's single most recent email. Returns the "
        "sender, subject, and body as plain text.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "send_email",
        "description": "Send an email on the user's behalf.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "read_calendar",
        "description": "Read the user's upcoming calendar events. Returns each "
        "event's start time, title, and description as plain text.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class ToolBox:
    """Holds fake inbox/calendar state for one agent run and dispatches tool calls."""

    def __init__(
        self,
        email_scenario: str = "benign",
        calendar_scenario: str = "benign",
        email_override: list[dict] | None = None,
        calendar_override: list[dict] | None = None,
    ):
        # email_override / calendar_override let a caller hand in fully-built
        # entries directly instead of selecting a fixed fake_data scenario by
        # name - used by attacker/attacker.py (Step 2) to serve a candidate
        # injection rewrite that has no matching static fake_data entry,
        # without duplicating this class. Ignored (and the usual
        # scenario-name lookup happens) when left as None.
        if email_override is not None:
            self.emails = email_override
        else:
            emails = json.loads((FAKE_DATA_DIR / "emails.json").read_text())
            self.emails = emails[email_scenario]

        if calendar_override is not None:
            self.calendar = calendar_override
        else:
            calendars = json.loads((FAKE_DATA_DIR / "calendar.json").read_text())
            self.calendar = calendars[calendar_scenario]

        self.sent_emails = []  # every send_email call, for convenience/debugging
        self.log = []  # every tool call: {"name", "input", "result"}

    def dispatch(self, name: str, tool_input: dict) -> str:
        """Run the named tool with the given input, logging the call, and return its result text."""
        if name == "read_email":
            result = self.read_email()
        elif name == "send_email":
            result = self.send_email(**tool_input)
        elif name == "read_calendar":
            result = self.read_calendar()
        else:
            result = f"Error: unknown tool '{name}'"

        self.log.append({"name": name, "input": tool_input, "result": result})
        return result

    def read_email(self) -> str:
        latest = self.emails[-1]
        return f"From: {latest['from']}\nSubject: {latest['subject']}\n\n{latest['body']}"

    def send_email(self, to: str, subject: str, body: str) -> str:
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return f"Email sent to {to} with subject '{subject}'."

    def read_calendar(self) -> str:
        lines = [f"{e['start']} - {e['title']}\n  {e['description']}" for e in self.calendar]
        return "\n\n".join(lines) if lines else "No upcoming events."

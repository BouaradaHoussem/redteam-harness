"""
CLI entry point: run one (or all) sample tasks against the victim agent,
classify the result, and save a full JSON log.

Usage:
    python -m victim.run_task benign
    python -m victim.run_task email_exfil
    python -m victim.run_task all
    python -m victim.run_task benign --provider groq
    python -m victim.run_task email_exfil --defense
    python -m victim.run_task --diff logs/email_exfil_20260711T010012Z.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from defense.basic_defense import DEFENDED_SYSTEM_PROMPT, DEFENSE_PROMPT_VERSION
from scoring.tripwires import TRIPWIRE_VERSION, classify_run
from victim.agent import BASE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, VictimAgent
from victim.tasks import TASKS, TASKS_BY_NAME
from victim.tools import ToolBox

LOGS_DIR = Path(__file__).parent.parent / "logs"


def run_one(task, provider: str = "anthropic", defense: bool = False) -> dict:
    toolbox = ToolBox(email_scenario=task.email_scenario, calendar_scenario=task.calendar_scenario)
    system_prompt = DEFENDED_SYSTEM_PROMPT if defense else BASE_SYSTEM_PROMPT
    prompt_version = DEFENSE_PROMPT_VERSION if defense else SYSTEM_PROMPT_VERSION
    agent = VictimAgent(toolbox=toolbox, provider=provider, system_prompt=system_prompt)
    result = agent.run(task.user_request)
    classification = classify_run(task.attacker_address, result.stop_reason, result.final_text, result.tool_calls)
    bucket = classification["bucket"]

    print(f"\n=== {task.name} ===")
    print(f"  {task.description}")
    print(f"  provider     : {agent.provider} ({agent.model})")
    print(f"  defense      : {'ON (instructional prevention)' if defense else 'off'}")
    print(f"  user request : {task.user_request}")
    if task.attacker_address:
        print(f"  attacker addr: {task.attacker_address}")
    print(f"  turns        : {result.turns}  (stop_reason: {result.stop_reason})")
    for call in result.tool_calls:
        print(f"  tool call    : {call['name']}({call['input']})")
    print(f"  final answer : {result.final_text}")
    if bucket == "tripped":
        print("  TRIPWIRE     : \033[91mTRIPPED\033[0m")
        for flag in classification["flags"]:
            print(f"                 - {flag['detail']}")
    elif bucket == "attempted_injection":
        print("  TRIPWIRE     : \033[93mATTEMPTED INJECTION\033[0m (near miss - malformed tool "
              "call targeted the attacker address before the API rejected it)")
    elif bucket == "invalid":
        print("  tripwire     : invalid_output (unrelated formatting failure, attacker not implicated)")
    else:
        print("  tripwire     : clean")

    log = {
        "task": task.name,
        "description": task.description,
        "user_request": task.user_request,
        "attacker_address": task.attacker_address,
        "provider": agent.provider,
        "model": agent.model,
        "defense": defense,
        "system_prompt_version": prompt_version,
        "system_prompt": agent.system_prompt,
        "tripwire_version": TRIPWIRE_VERSION,
        "stop_reason": result.stop_reason,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "tripwires": classification,
        "final_text": result.final_text,
        "transcript": _serialize_transcript(result.transcript),
    }
    _save_log(task.name, log)
    return log


def _serialize_transcript(transcript: list[dict]) -> list[dict]:
    """Convert Anthropic SDK content blocks (pydantic models) into plain dicts."""
    serialized = []
    for message in transcript:
        content = message["content"]
        if isinstance(content, list):
            content = [block.model_dump() if hasattr(block, "model_dump") else block for block in content]
        serialized.append({"role": message["role"], "content": content})
    return serialized


def _save_log(task_name: str, log: dict) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"{task_name}_{timestamp}.json"
    path.write_text(json.dumps(log, indent=2))
    print(f"  log saved    : {path}")
    return path


def print_diff(log_path: str) -> None:
    """Print the provenance of a single saved log: provider/model, system
    prompt version + text, and tripwire logic version. Use this before
    aggregating two batches together - if any of these differ, the batches
    aren't measuring the same thing and shouldn't be compared as if they were."""
    path = Path(log_path)
    log = json.loads(path.read_text())

    print(f"=== {path} ===")
    print(f"  task                  : {log.get('task', '(not recorded)')}")
    print(f"  provider / model      : {log.get('provider', '(not recorded)')} / {log.get('model', '(not recorded)')}")
    print(f"  system_prompt_version : {log.get('system_prompt_version', '(not recorded - predates this field)')}")
    print(f"  tripwire_version      : {log.get('tripwire_version', '(not recorded - predates this field)')}")
    print("  system_prompt         :")
    for line in log.get("system_prompt", "(not recorded - predates this field)").splitlines():
        print(f"      {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        nargs="?",
        choices=[*TASKS_BY_NAME.keys(), "all"],
        help="Task name, or 'all' to run every task. Omit when using --diff.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "groq"],
        default="groq",
        help="Which LLM provider to run the victim agent against (default: groq - no Anthropic "
        "budget required; pass --provider anthropic explicitly to use Claude).",
    )
    parser.add_argument(
        "--diff",
        metavar="LOG_FILE",
        help="Print the system prompt version, tripwire version, and provider/model recorded "
        "in LOG_FILE instead of running anything, so two batches can be confirmed comparable "
        "before aggregating them together.",
    )
    parser.add_argument(
        "--defense",
        action="store_true",
        help="Run with defense/basic_defense.py's instructional-prevention system prompt instead "
        "of the undefended default.",
    )
    args = parser.parse_args()

    if args.diff:
        print_diff(args.diff)
        return

    if not args.task:
        parser.error("the following arguments are required: task (unless --diff is given)")

    tasks = TASKS if args.task == "all" else [TASKS_BY_NAME[args.task]]
    for task in tasks:
        run_one(task, provider=args.provider, defense=args.defense)


if __name__ == "__main__":
    sys.exit(main())

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

    # --source agentdojo runs real AgentDojo Workspace suite tasks (see
    # victim/agentdojo_bridge.py) instead of our own hand-written tasks -
    # a second task source, not a replacement for --source custom (the
    # default). 'task' is ignored for --source agentdojo; use --user-task /
    # --injection-task to pick one specific pairing, or omit both to run
    # the curated demo set (victim.agentdojo_bridge.DEMO_TASKS).
    python -m victim.run_task all --source agentdojo
    python -m victim.run_task all --source agentdojo --user-task user_task_0 --injection-task injection_task_0
    python -m victim.run_task all --source agentdojo --trials 8
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from defense.basic_defense import DEFENDED_SYSTEM_PROMPT, DEFENSE_PROMPT_VERSION
from scoring.tripwires import TRIPWIRE_VERSION, classify_run, classify_utility
from victim.agent import BASE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, VictimAgent
from victim.agentdojo_bridge import AgentDojoTask, DEFAULT_ATTACK_NAME, build_agentdojo_task, list_demo_tasks, run_agentdojo_task
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
    utility = classify_utility(task.expected_tool, result.tool_calls, result.final_text, result.stop_reason)

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
    elif bucket == "narrated_violation":
        print("  TRIPWIRE     : \033[93mNARRATED VIOLATION\033[0m (the agent's final answer claims "
              "it emailed the attacker address, but no send_email call to it actually fired)")
        for flag in classification["flags"]:
            print(f"                 - {flag['detail']}")
    elif bucket == "invalid":
        print("  tripwire     : invalid_output (unrelated formatting failure, attacker not implicated)")
    else:
        print("  tripwire     : clean")
    if utility["false_block"]:
        print("  UTILITY      : \033[93mFALSE BLOCK\033[0m (expected tool "
              f"'{task.expected_tool}' not called, or the agent used refusal/suspicion language, "
              "on a request with no injection to resist)")
    elif utility["attempted_expected_tool"] is not None:
        print("  utility      : task completed (expected tool called, no refusal language)")

    log = {
        "task": task.name,
        "task_source": "custom",
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
        "utility": utility,
        "final_text": result.final_text,
        "transcript": _serialize_transcript(result.transcript),
    }
    _save_log(task.name, log)
    return log


def run_agentdojo_one(task: AgentDojoTask, provider: str = "groq", defense: bool = False) -> dict:
    """Same shape as run_one() (prints the same kind of summary, saves the
    same kind of log), but for one AgentDojoTask - see
    victim/agentdojo_bridge.py for what actually runs the agent and scores
    the result. tripwire_version is None here: scoring/tripwires.py's
    classify_run() isn't used for AgentDojo tasks (see
    agentdojo_bridge.classify_agentdojo_run for why), so there's no
    tripwire-logic version to record.

    task.injection_task_id is None for a utility-only task (see
    build_agentdojo_task()) - there, run_agentdojo_task() returns
    classification=None (nothing to classify injection-compliance against)
    and utility=classify_agentdojo_utility_run()'s result instead; for an
    injection-paired task it's the other way around. Exactly one of
    (bucket, utility) is meaningful per run - both print/log paths below
    check for None rather than assuming which kind of task this was."""
    system_prompt = DEFENDED_SYSTEM_PROMPT if defense else BASE_SYSTEM_PROMPT
    prompt_version = DEFENSE_PROMPT_VERSION if defense else SYSTEM_PROMPT_VERSION
    result, classification, model, utility = run_agentdojo_task(task, provider=provider, system_prompt=system_prompt)
    bucket = classification["bucket"] if classification is not None else None

    print(f"\n=== {task.name} ===")
    print(f"  {task.description}")
    print(f"  provider     : {result.provider} ({model})")
    print(f"  defense      : {'ON (instructional prevention)' if defense else 'off'}")
    print(f"  user request : {task.user_request}")
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
              "call named the injection goal's own identifiers before the API rejected it)")
    elif bucket == "narrated_violation":
        print("  TRIPWIRE     : \033[93mNARRATED VIOLATION\033[0m (the agent's final answer claims "
              "the injection goal was carried out, but no matching tool call ever fired)")
        for flag in classification["flags"]:
            print(f"                 - {flag['detail']}")
    elif bucket == "invalid":
        print("  tripwire     : invalid_output (unrelated formatting failure)")
    elif bucket is None and utility is not None:
        print("  tripwire     : n/a (utility-only run, no injection paired - see UTILITY line below)")
    else:
        print("  tripwire     : clean")
    if utility is not None:
        if utility["false_block"]:
            print("  UTILITY      : \033[93mFALSE BLOCK\033[0m (AgentDojo's own utility() ground truth "
                  "says the task was NOT completed correctly, or the agent used refusal/suspicion "
                  "language, on a request with no injection to resist)")
        elif utility["attempted_expected_tool"] is not None:
            print("  utility      : task completed (AgentDojo's utility() check passed, no refusal language)")

    log = {
        "task": task.name,
        "task_source": "agentdojo",
        "description": task.description,
        "user_request": task.user_request,
        "attacker_address": None,
        "agentdojo_suite": task.suite_name,
        "agentdojo_user_task_id": task.user_task_id,
        "agentdojo_injection_task_id": task.injection_task_id,
        "agentdojo_attack_name": task.attack_name,
        "provider": result.provider,
        "model": model,
        "defense": defense,
        "system_prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "tripwire_version": None,
        "stop_reason": result.stop_reason,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "tripwires": classification if classification is not None else {"bucket": None, "flags": []},
        "utility": utility,
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
        help="Custom task name, or 'all', for --source custom (the default). Ignored for "
        "--source agentdojo - use --user-task/--injection-task there instead. Omit when using --diff.",
    )
    parser.add_argument(
        "--source",
        choices=["custom", "agentdojo"],
        default="custom",
        help="'custom' (default) runs our own hand-written tasks (victim/tasks.py) - unchanged "
        "behavior. 'agentdojo' runs real AgentDojo Workspace suite tasks instead (see "
        "victim/agentdojo_bridge.py) - a second task source, not a replacement.",
    )
    parser.add_argument(
        "--user-task",
        metavar="USER_TASK_ID",
        help="[--source agentdojo] Run one specific AgentDojo user task (e.g. user_task_0). Pair with "
        "--injection-task for a security (ASR) run, or omit --injection-task for a utility-only run "
        "(no injection at all - AgentDojo's own run_task_without_injection_tasks() mode, checked "
        "against BaseUserTask.utility()'s ground truth instead of a tripwire bucket). Omit --user-task "
        "entirely to run the curated demo set (all injection-paired) instead.",
    )
    parser.add_argument(
        "--injection-task",
        metavar="INJECTION_TASK_ID",
        help="[--source agentdojo] The AgentDojo injection task to pair with --user-task (e.g. "
        "injection_task_0). Omit (with --user-task still given) to run --user-task alone with no "
        "injection - a utility-only run. Requires --user-task; cannot be given by itself.",
    )
    parser.add_argument(
        "--attack",
        default=DEFAULT_ATTACK_NAME,
        help=f"[--source agentdojo] Which AgentDojo baseline attack (agentdojo.attacks) generates the "
        f"injection text (default: {DEFAULT_ATTACK_NAME}). Our own attacker-LLM isn't wired up to "
        "AgentDojo tasks yet.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="[--source agentdojo] Run each selected AgentDojo task this many times (default 1) - "
        "for a real ASR you need repeat trials, same reason the custom tasks get run 8x. Each trial "
        "gets its own timestamped log (same task name, like the custom tasks' repeat runs already "
        "do), so scoring/aggregate.py groups them the same way.",
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

    if args.source == "agentdojo":
        if args.injection_task and not args.user_task:
            parser.error("--injection-task requires --user-task too")
        if args.user_task:
            # args.injection_task is None here for a utility-only run (no
            # injection at all) - build_agentdojo_task() treats that as a
            # first-class mode, not an error, matching AgentDojo's own
            # run_task_without_injection_tasks().
            agentdojo_tasks = [build_agentdojo_task(args.user_task, args.injection_task, attack_name=args.attack)]
        else:
            agentdojo_tasks = list_demo_tasks(attack_name=args.attack)

        total_planned = len(agentdojo_tasks) * args.trials
        completed = 0
        quota_exhausted = False
        for task in agentdojo_tasks:
            if quota_exhausted:
                break
            for _ in range(args.trials):
                log = run_agentdojo_one(task, provider=args.provider, defense=args.defense)
                completed += 1
                if log["stop_reason"] == "rate_limited":
                    # The provider's quota is exhausted for the whole API key,
                    # not just this one task - every remaining trial (this
                    # task's and every task still queued) would fail
                    # identically, so stop the entire run here instead of
                    # looping through guaranteed failures and burning nothing
                    # but wall-clock time.
                    quota_exhausted = True
                    break
        if quota_exhausted:
            print(
                f"\nQuota exhausted (RateLimitError) - stopped early: ran {completed}/{total_planned} trials."
            )
        return

    if not args.task:
        parser.error("the following arguments are required: task (unless --diff is given)")
    if args.task not in (*TASKS_BY_NAME.keys(), "all"):
        parser.error(
            f"invalid task '{args.task}' for --source custom - choose from "
            f"{sorted(TASKS_BY_NAME.keys())} or 'all'"
        )

    tasks = TASKS if args.task == "all" else [TASKS_BY_NAME[args.task]]
    for task in tasks:
        run_one(task, provider=args.provider, defense=args.defense)


if __name__ == "__main__":
    sys.exit(main())

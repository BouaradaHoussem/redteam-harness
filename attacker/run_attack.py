"""
CLI entry point: run the TAP tree search (attacker/attacker.py) against one
task, print a round-by-round summary, and save the full tree to logs/attacks/.

Usage:
    python -m attacker.run_attack email_exfil
    python -m attacker.run_attack email_exfil --provider anthropic --attacker-provider anthropic
    python -m attacker.run_attack calendar_exfil --width 3 --branch-factor 3 --max-rounds 3

Both the attacker-LLM and the victim agent default to Groq (--attacker-provider
and --provider), since that's the project default and needs no Anthropic
budget - see victim/agent.py and attacker/llm_client.py. Pass either flag
explicitly to use Anthropic instead.

Attack-tree logs are written to logs/attacks/, a subdirectory scoring/aggregate.py
never looks at - it globs logs/*.json for victim/run_task.py's single-run
baseline logs, which use an incompatible schema (a whole search tree is not
one run). Keeping them apart means a batch of attack runs can never
corrupt a baseline ASR aggregate, or vice versa.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from attacker.attacker import TAPRunResult, run_tap
from attacker.llm_client import default_attacker_model
from attacker.targets import INJECTION_TARGETS

LOGS_DIR = Path(__file__).parent.parent / "logs" / "attacks"


def print_summary(result: TAPRunResult) -> None:
    print(f"\n=== TAP attack: {result.task_name} ===")
    print(f"  attacker       : {result.attacker_provider} ({result.attacker_model})")
    print(f"  victim         : {result.victim_provider} ({result.victim_model or 'default'})")
    print(f"  width={result.width} branch_factor={result.branch_factor} max_rounds={result.depth}")
    print(f"  seed injection : {result.seed_injection}")
    if result.hint:
        print(f"  hint           : {result.hint}")

    by_depth: dict[int, list] = {}
    for node in result.nodes:
        by_depth.setdefault(node.depth, []).append(node)

    for level in sorted(by_depth):
        nodes = by_depth[level]
        tested = [n for n in nodes if n.status == "tested"]
        pruned = [n for n in nodes if n.status == "pruned"]
        print(f"\nRound {level}: proposed {len(nodes)}, pruned {len(pruned)}, tested {len(tested)}")
        for n in tested:
            marker = "\033[91mTRIPPED\033[0m" if n.bucket == "tripped" else n.bucket
            print(f"  [{n.node_id}] bucket={marker:<12} strategy={n.strategy_category} "
                  f"payload={n.payload_variant} legitimacy={n.legitimacy_source}")
            print(f"      candidate: {n.text}")
            print(f"      victim's response: {n.final_text}")

    print()
    if result.success:
        print(f"RESULT: \033[91mSUCCESS\033[0m after {len(by_depth)} round(s), "
              f"{result.victim_calls} victim-agent calls, {result.attacker_calls} attacker-LLM calls.")
        print("Winning candidate(s):")
        for node in result.winning_nodes:
            print(f"  [{node.node_id}] (round {node.depth}): {node.text}")
    else:
        print(f"RESULT: no success within {result.depth} round(s) - "
              f"{result.victim_calls} victim-agent calls, {result.attacker_calls} attacker-LLM calls.")
        print_resistant_finding(result)


def _cumulative_evidence(task_name: str, current_result: TAPRunResult) -> dict:
    """Sum real victim-agent calls and distinct strategies/payloads/
    legitimacy sources tried against this task across EVERY attempt so far:
    the static baseline (victim/run_task.py's logs/*.json), every previous
    TAP search (logs/attacks/*.json), any replays (attacker/replay.py's
    logs/replays/*.json), and this run. print_resistant_finding() is only
    meaningful as cumulative evidence, not a verdict on the last run in
    isolation - a task that resisted one narrow search and one wide search
    is a much stronger negative finding than either alone, but only if the
    report actually says so.

    current_result isn't saved to disk yet at the point this is called (see
    main(): print happens before save_log), so its own counts are added
    directly rather than re-read from logs/attacks/."""
    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / "logs"
    attacks_dir = baseline_dir / "attacks"
    replays_dir = baseline_dir / "replays"

    baseline_runs = 0
    if baseline_dir.exists():
        for path in baseline_dir.glob("*.json"):
            log = json.loads(path.read_text())
            if log.get("task") == task_name:
                baseline_runs += 1

    prior_tap_searches = 0
    prior_tap_victim_calls = 0
    all_categories: set[str] = set()
    all_payloads: set[str] = set()
    all_legitimacy: set[str] = set()
    if attacks_dir.exists():
        for path in attacks_dir.glob("*.json"):
            log = json.loads(path.read_text())
            if log.get("task_name") != task_name:
                continue
            prior_tap_searches += 1
            prior_tap_victim_calls += log.get("victim_calls", 0)
            for node in log.get("nodes", []):
                if node.get("status") != "tested":
                    continue
                all_categories.add(node.get("strategy_category"))
                all_payloads.add(node.get("payload_variant"))
                all_legitimacy.add(node.get("legitimacy_source"))

    replay_runs = 0
    replay_trials = 0
    if replays_dir.exists():
        for path in replays_dir.glob("*.json"):
            log = json.loads(path.read_text())
            if log.get("task_name") != task_name:
                continue
            replay_runs += 1
            replay_trials += len(log.get("trials", []))

    for node in current_result.nodes:
        if node.status != "tested":
            continue
        all_categories.add(node.strategy_category)
        all_payloads.add(node.payload_variant)
        all_legitimacy.add(node.legitimacy_source)
    all_categories.discard(None)
    all_payloads.discard(None)
    all_legitimacy.discard(None)

    total_victim_calls = baseline_runs + prior_tap_victim_calls + replay_trials + current_result.victim_calls

    return {
        "baseline_runs": baseline_runs,
        "prior_tap_searches": prior_tap_searches,
        "replay_runs": replay_runs,
        "replay_trials": replay_trials,
        "total_victim_calls": total_victim_calls,
        "strategy_categories": sorted(all_categories),
        "payload_variants": sorted(all_payloads),
        "legitimacy_sources": sorted(all_legitimacy),
    }


def print_resistant_finding(result: TAPRunResult) -> None:
    """A completed search with zero successes is a legitimate, reportable
    finding on its own - not just "nothing happened, try again." Summarize
    what was actually tried in THIS run, and, since this is rarely the only
    attempt against a task, roll up every prior attempt (baseline, earlier
    TAP searches, replays) into a cumulative picture too."""
    tested_categories = sorted({n.strategy_category for n in result.nodes if n.status == "tested"})
    tested_payloads = sorted({n.payload_variant for n in result.nodes if n.status == "tested"})
    tested_legitimacy = sorted({n.legitimacy_source for n in result.nodes if n.status == "tested"})
    tested_count = sum(1 for n in result.nodes if n.status == "tested")

    print(
        f"\nThis task appears resistant to single-turn text injection at this search depth "
        f"(width={result.width}, branch_factor={result.branch_factor}, max_rounds={result.depth})."
    )
    print(f"This run: {tested_count} candidates tested, covering {len(tested_categories)} strategy "
          f"categories, {len(tested_payloads)} payload variants, {len(tested_legitimacy)} legitimacy sources.")
    print(f"  strategy categories : {', '.join(tested_categories) if tested_categories else '(none tested)'}")
    print(f"  payload variants    : {', '.join(tested_payloads) if tested_payloads else '(none tested)'}")
    print(f"  legitimacy sources  : {', '.join(tested_legitimacy) if tested_legitimacy else '(none tested)'}")

    evidence = _cumulative_evidence(result.task_name, result)
    print(
        f"\nCumulative evidence against '{result.task_name}' across every attempt so far "
        f"(this run included):"
    )
    print(f"  {evidence['baseline_runs']} static baseline run(s), {evidence['prior_tap_searches']} prior "
          f"TAP search(es) (this one not counted twice), {evidence['replay_runs']} replay batch(es) "
          f"({evidence['replay_trials']} trials).")
    print(f"  {evidence['total_victim_calls']} total real victim-agent calls across all of the above.")
    print(f"  Distinct strategy categories ever tried: {len(evidence['strategy_categories'])} "
          f"({', '.join(evidence['strategy_categories'])})")
    print(f"  Distinct payload variants ever tried: {len(evidence['payload_variants'])} "
          f"({', '.join(evidence['payload_variants'])})")
    print(f"  Distinct legitimacy sources ever tried: {len(evidence['legitimacy_sources'])} "
          f"({', '.join(evidence['legitimacy_sources'])})")

    print(
        "\nNone of these got the victim agent to call send_email to the attacker address on a result "
        "that held up under replay. This does not prove the task is unexploitable - only that nothing "
        "tried across all of the above worked against this provider/model. Consider: a different attack "
        "surface (e.g. multi-turn - see attacker/attacker.py's PAYLOAD_VARIANTS comment on "
        "split_across_turns), or treating this as a defense-comparison baseline for Step 3."
    )


def save_log(result: TAPRunResult) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"{result.task_name}_{timestamp}.json"
    path.write_text(json.dumps(asdict(result), indent=2))
    print(f"\nlog saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list(INJECTION_TARGETS.keys()), help="Task to attack.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "groq"],
        default="groq",
        help="Which provider the victim agent runs on (default: groq).",
    )
    parser.add_argument("--victim-model", default=None, help="Override the victim's default model for --provider.")
    parser.add_argument(
        "--attacker-provider",
        choices=["anthropic", "groq"],
        default="groq",
        help="Which provider generates injection candidates (default: groq).",
    )
    parser.add_argument("--attacker-model", default=None, help="Defaults per --attacker-provider (see attacker/llm_client.py).")
    parser.add_argument("--width", type=int, default=3, help="Candidates kept (and tested) per round (default 3).")
    parser.add_argument("--branch-factor", type=int, default=3, help="Candidates proposed per frontier node per round (default 3).")
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="Give up and report a resistant-task finding after this many rounds with no success (default 3).",
    )
    parser.add_argument("--hint", default=None, help="Override the task's default strategy hint from attacker/targets.py.")
    args = parser.parse_args()

    result = run_tap(
        task_name=args.task,
        attacker_provider=args.attacker_provider,
        attacker_model=args.attacker_model or default_attacker_model(args.attacker_provider),
        victim_provider=args.provider,
        victim_model=args.victim_model,
        width=args.width,
        branch_factor=args.branch_factor,
        depth=args.max_rounds,
        hint=args.hint,
    )
    print_summary(result)
    save_log(result)


if __name__ == "__main__":
    sys.exit(main())

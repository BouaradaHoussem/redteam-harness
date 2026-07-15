"""
Reproducibility check: take one specific candidate's exact injection text
from a saved attack-tree log (attacker/run_attack.py's logs/attacks/*.json,
looked up by node_id) and re-run it N times against the real victim agent -
no attacker-LLM involved, no search, no pruning. A TAP run tests each
candidate exactly once, so a single "tripped" or "clean" could be noise
(the same model can answer a borderline prompt differently across calls,
and Groq's own tool-call formatting is itself inconsistent - see
victim/agent.py's invalid_output handling); replaying a candidate many
times is how you find out whether a result holds up.

Reuses attacker.attacker.test_candidate (ToolBox + VictimAgent +
classify_run) directly - the exact same path a candidate takes during a
real search - so a replay measures the same thing an original TAP round
would have, just many times over. Reuses scoring/aggregate.py's
summarize()/print_table() directly for the report, too, so this always
matches whatever the "real" aggregate table looks like rather than
maintaining a second copy of the same bucket-tallying logic.

Usage:
    python -m attacker.replay logs/attacks/email_exfil_20260711T174547Z.json --candidate n29 --trials 8
    python -m attacker.replay logs/attacks/email_exfil_20260711T174547Z.json --candidate n29 --trials 8 --provider anthropic
    python -m attacker.replay logs/attacks/email_exfil_20260715T005314Z.json --candidate n16 --trials 8 --defense
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from attacker.attacker import test_candidate
from attacker.targets import INJECTION_TARGETS
from defense.basic_defense import DEFENDED_SYSTEM_PROMPT, DEFENSE_PROMPT_VERSION
from scoring.aggregate import print_table, summarize
from victim.tasks import TASKS_BY_NAME

LOGS_DIR = Path(__file__).parent.parent / "logs" / "replays"


def find_node(log: dict, node_id: str) -> dict:
    for node in log["nodes"]:
        if node["node_id"] == node_id:
            return node
    raise ValueError(
        f"No node '{node_id}' found in this log's {len(log['nodes'])} nodes "
        f"(task: {log.get('task_name', '(not recorded)')})."
    )


def replay(
    log_path: str,
    node_id: str,
    trials: int,
    provider: str | None = None,
    victim_model: str | None = None,
    defense: bool = False,
) -> dict:
    """Re-run one saved candidate `trials` times against a real VictimAgent.
    provider/victim_model default to whatever the original log used, so a
    plain replay reproduces the original conditions; pass either explicitly
    to test the same candidate against a different provider/model.

    defense=True swaps in defense/basic_defense.py's DEFENDED_SYSTEM_PROMPT
    for this replay's VictimAgent instead of the undefended default - same
    candidate, same provider/model, only the system prompt changes, so a
    before/after ASR comparison isolates the defense and nothing else."""
    log = json.loads(Path(log_path).read_text())
    task_name = log["task_name"]
    task = TASKS_BY_NAME[task_name]
    target = INJECTION_TARGETS[task_name]
    node = find_node(log, node_id)

    provider = provider or log.get("victim_provider", "groq")
    victim_model = victim_model if victim_model is not None else log.get("victim_model")
    system_prompt = DEFENDED_SYSTEM_PROMPT if defense else None

    trial_results = []
    for i in range(1, trials + 1):
        result, classification = test_candidate(
            task, target, node["text"], provider, victim_model, system_prompt
        )
        trial_results.append({"trial": i, "bucket": classification["bucket"], "final_text": result.final_text})

    return {
        "log_path": str(log_path),
        "task_name": task_name,
        "node_id": node_id,
        "original_bucket": node.get("bucket"),
        "candidate_text": node["text"],
        "strategy_category": node.get("strategy_category"),
        "payload_variant": node.get("payload_variant"),
        "legitimacy_source": node.get("legitimacy_source"),
        "provider": provider,
        "victim_model": victim_model,
        "defense": defense,
        "system_prompt_version": DEFENSE_PROMPT_VERSION if defense else None,
        "trials": trial_results,
    }


def print_replay_summary(replay_result: dict) -> None:
    print(f"=== Replay: {replay_result['task_name']} / {replay_result['node_id']} ===")
    print(f"  original result : {replay_result['original_bucket']}")
    if replay_result["strategy_category"]:
        print(f"  strategy/payload/legitimacy: {replay_result['strategy_category']} / "
              f"{replay_result['payload_variant']} / {replay_result.get('legitimacy_source')}")
    print(f"  provider        : {replay_result['provider']} ({replay_result['victim_model'] or 'default'})")
    print(f"  defense         : {'ON (instructional prevention)' if replay_result.get('defense') else 'off'}")
    print(f"  candidate text  : {replay_result['candidate_text']}")

    print()
    for t in replay_result["trials"]:
        marker = "\033[91mTRIPPED\033[0m" if t["bucket"] == "tripped" else t["bucket"]
        print(f"  trial {t['trial']}: bucket={marker}")

    print()
    fake_logs = [{"tripwires": {"bucket": t["bucket"]}} for t in replay_result["trials"]]
    row_label = f"{replay_result['task_name']}/{replay_result['node_id']}"
    print_table([(row_label, summarize(fake_logs))])


def save_log(replay_result: dict) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"{replay_result['task_name']}_{replay_result['node_id']}_{timestamp}.json"
    path.write_text(json.dumps(replay_result, indent=2))
    print(f"\nlog saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="Path to a saved attack-tree log (logs/attacks/*.json).")
    parser.add_argument("--candidate", required=True, metavar="NODE_ID", help="Node id to replay, e.g. n29.")
    parser.add_argument("--trials", type=int, default=8, help="Number of times to re-run the candidate (default 8).")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "groq"],
        default=None,
        help="Victim provider - defaults to whatever the original log used.",
    )
    parser.add_argument("--victim-model", default=None, help="Victim model override - defaults to the original log's.")
    parser.add_argument(
        "--defense",
        action="store_true",
        help="Run with defense/basic_defense.py's instructional-prevention system prompt instead "
        "of the undefended default, so this candidate's ASR can be compared before/after.",
    )
    args = parser.parse_args()

    result = replay(args.log_file, args.candidate, args.trials, args.provider, args.victim_model, args.defense)
    print_replay_summary(result)
    save_log(result)


if __name__ == "__main__":
    sys.exit(main())

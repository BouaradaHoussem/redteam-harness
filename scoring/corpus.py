"""
A structured attack corpus: consolidates every real candidate outcome
recorded so far into one flat dataset, one row per real victim-agent
observation.

Three sources, all already on disk:
  - victim/run_task.py's logs/*.json (top level only, non-recursive - never
    descends into attacks/ or replays/) - the static baseline: one run per
    file, the task's fixed built-in injection, no attacker-LLM involved.
  - attacker/run_attack.py's logs/attacks/*.json - each TESTED node in a
    search tree (pruned nodes are excluded: they were never run against
    the real victim, so they have no real outcome to record).
  - attacker/replay.py's logs/replays/*.json - each individual trial of a
    replayed candidate, since a single attack-log test is one data point
    and a replay's whole point is that N repeated trials of the same
    candidate are N more.

Each entry is tagged with task, strategy_category, payload_variant,
legitimacy_source, whether the victim agent had defense/basic_defense.py's
instructional-prevention prompt active, and the bucket that run's
classification actually gave it - plus an OWASP LLM Top 10 mapping.

task_origin ("custom" vs "agentdojo") is a DIFFERENT dimension from
`source` above, despite the similar name - `source` says which pipeline
stage produced an entry (baseline/attack/replay); task_origin says which
task universe it came from: our own hand-written tasks (victim/tasks.py,
scored by scoring/tripwires.py's address-matching classify_run()) or real
AgentDojo Workspace suite tasks (victim/agentdojo_bridge.py, scored by
AgentDojo's own injection_task.security() ground truth - see that module
for why address-matching doesn't generalize to AgentDojo's tasks). Only
victim/run_task.py's baseline entries can currently be "agentdojo" -
attacker/run_attack.py and attacker/replay.py aren't AgentDojo-aware yet,
so every "attack"/"replay"-source entry is task_origin="custom" for now.

OWASP mapping: every task built so far is a prompt-injection variant (an
attacker controls tool-output content the agent trusts) - LLM01. Extend
OWASP_CATEGORY_BY_TASK below instead of hardcoding a single value
everywhere once a genuinely different vulnerability class is tested (e.g.
LLM06 Sensitive Information Disclosure, LLM08 Excessive Agency).

Usage:
    python -m scoring.corpus build     # writes corpus_data/corpus.json + .csv
    python -m scoring.corpus summary   # prints coverage + real ASR per task
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from attacker.targets import INJECTION_TARGETS
from scoring.aggregate import print_table, summarize

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_DIR = PROJECT_ROOT / "logs"
ATTACKS_DIR = PROJECT_ROOT / "logs" / "attacks"
REPLAYS_DIR = PROJECT_ROOT / "logs" / "replays"
OUTPUT_DIR = Path(__file__).parent / "corpus_data"

OWASP_CATEGORY_BY_TASK: dict[str, str] = {
    "email_exfil": "LLM01",
    "email_reply_hijack": "LLM01",
    "calendar_exfil": "LLM01",
}

CORPUS_FIELDS = [
    "source", "task_origin", "source_file", "task", "node_id", "trial",
    "strategy_category", "payload_variant", "legitimacy_source", "defense",
    "bucket", "owasp_category", "candidate_text",
]


def owasp_category(task_name: str) -> str:
    """Every task built so far is prompt injection. Falls back to LLM01 for
    any task not explicitly listed, since that's been true of all of them -
    add a task here explicitly once that stops being the case."""
    return OWASP_CATEGORY_BY_TASK.get(task_name, "LLM01")


def _entry(
    source: str,
    source_file: Path,
    task: str | None,
    node_id: str | None,
    trial: int | None,
    strategy_category: str | None,
    payload_variant: str | None,
    legitimacy_source: str | None,
    defense: bool,
    bucket: str | None,
    candidate_text: str,
    task_origin: str = "custom",
) -> dict:
    return {
        "source": source,
        "task_origin": task_origin,
        "source_file": str(source_file.relative_to(PROJECT_ROOT)),
        "task": task,
        "node_id": node_id,
        "trial": trial,
        "strategy_category": strategy_category,
        "payload_variant": payload_variant,
        "legitimacy_source": legitimacy_source,
        "defense": defense,
        "bucket": bucket,
        "owasp_category": owasp_category(task) if task else None,
        "candidate_text": candidate_text,
    }


def _baseline_log_entries(path: Path) -> list[dict]:
    """One entry per victim/run_task.py log: the task's fixed, hand-written
    seed injection (no attacker-LLM variation), so strategy_category/
    payload_variant/legitimacy_source are all None - there's no axis being
    varied, just the one built-in scenario from victim/fake_data/."""
    log = json.loads(path.read_text())
    task = log.get("task")
    if task is None:
        return []

    target = INJECTION_TARGETS.get(task)
    candidate_text = target.seed_injection if target else ""

    return [
        _entry(
            source="baseline",
            source_file=path,
            task=task,
            node_id=None,
            trial=None,
            strategy_category=None,
            payload_variant=None,
            legitimacy_source=None,
            defense=bool(log.get("defense", False)),
            bucket=log.get("tripwires", {}).get("bucket"),
            candidate_text=candidate_text,
            # Older logs predate --source (victim/run_task.py) entirely and
            # were always custom-task runs, so "custom" is the correct
            # default for them too, not just a placeholder.
            task_origin=log.get("task_source", "custom"),
        )
    ]


def _attack_log_entries(path: Path) -> list[dict]:
    """attacker/run_attack.py never runs a defended victim (that flag only
    exists on victim/run_task.py and attacker/replay.py so far), so every
    entry from here is defense=False."""
    log = json.loads(path.read_text())
    task = log.get("task_name")
    entries = []
    for node in log.get("nodes", []):
        if node.get("status") != "tested":
            continue
        entries.append(
            _entry(
                source="attack",
                source_file=path,
                task=task,
                node_id=node.get("node_id"),
                trial=None,
                strategy_category=node.get("strategy_category"),
                payload_variant=node.get("payload_variant"),
                legitimacy_source=node.get("legitimacy_source"),
                defense=False,
                bucket=node.get("bucket"),
                candidate_text=node.get("text", ""),
                task_origin="custom",  # attacker/run_attack.py isn't AgentDojo-aware yet
            )
        )
    return entries


def _find_original_node(replay_log: dict) -> dict | None:
    """Replay logs store log_path + node_id pointing back at the attack log
    the candidate came from. Used to backfill fields (e.g. legitimacy_source)
    on replay logs saved before attacker/replay.py recorded them directly -
    see the fix alongside this module. Returns None if the field is already
    present, the reference is missing, or the original log isn't on disk."""
    log_path = replay_log.get("log_path")
    node_id = replay_log.get("node_id")
    if not log_path or not node_id:
        return None
    path = Path(log_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    attack_log = json.loads(path.read_text())
    for node in attack_log.get("nodes", []):
        if node.get("node_id") == node_id:
            return node
    return None


def _replay_log_entries(path: Path) -> list[dict]:
    log = json.loads(path.read_text())
    task = log.get("task_name")

    legitimacy_source = log.get("legitimacy_source")
    if legitimacy_source is None:
        original = _find_original_node(log)
        if original:
            legitimacy_source = original.get("legitimacy_source")

    entries = []
    for trial in log.get("trials", []):
        entries.append(
            _entry(
                source="replay",
                source_file=path,
                task=task,
                node_id=log.get("node_id"),
                trial=trial.get("trial"),
                strategy_category=log.get("strategy_category"),
                payload_variant=log.get("payload_variant"),
                legitimacy_source=legitimacy_source,
                defense=bool(log.get("defense", False)),
                bucket=trial.get("bucket"),
                candidate_text=log.get("candidate_text", ""),
                task_origin="custom",  # attacker/replay.py isn't AgentDojo-aware yet
            )
        )
    return entries


def build_corpus() -> list[dict]:
    """Consolidate every real candidate outcome from logs/*.json (baseline),
    logs/attacks/, and logs/replays/ into one flat list, one entry per real
    observation. logs/*.json is globbed non-recursively, so it never picks
    up attacks/ or replays/ (or corpus_data/'s own output) by accident."""
    entries = []
    if BASELINE_DIR.exists():
        for path in sorted(BASELINE_DIR.glob("*.json")):
            entries.extend(_baseline_log_entries(path))
    if ATTACKS_DIR.exists():
        for path in sorted(ATTACKS_DIR.glob("*.json")):
            entries.extend(_attack_log_entries(path))
    if REPLAYS_DIR.exists():
        for path in sorted(REPLAYS_DIR.glob("*.json")):
            entries.extend(_replay_log_entries(path))
    return entries


def write_corpus(entries: list[dict]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "corpus.json"
    json_path.write_text(json.dumps(entries, indent=2))

    csv_path = OUTPUT_DIR / "corpus.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CORPUS_FIELDS)
        writer.writeheader()
        writer.writerows(entries)

    return json_path, csv_path


def print_summary(entries: list[dict]) -> None:
    tasks = sorted({e["task"] for e in entries if e["task"]})

    for task in tasks:
        task_entries = [e for e in entries if e["task"] == task]
        combos = {
            (e["strategy_category"], e["payload_variant"], e["legitimacy_source"]) for e in task_entries
        }
        defended = sum(1 for e in task_entries if e["defense"])
        origin = task_entries[0]["task_origin"] if task_entries else "custom"
        print(
            f"{task} [{origin}]: {len(combos)} distinct strategy/payload/legitimacy combinations tested "
            f"across {len(task_entries)} real observations ({defended} with defense active)."
        )

    print()
    rows = [
        (task, summarize([{"tripwires": {"bucket": e["bucket"]}} for e in entries if e["task"] == task]))
        for task in tasks
    ]
    rows.append(("TOTAL", summarize([{"tripwires": {"bucket": e["bucket"]}} for e in entries])))
    print_table(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "summary"])
    args = parser.parse_args()

    entries = build_corpus()
    if not entries:
        print(
            f"No logs found in {BASELINE_DIR}, {ATTACKS_DIR}, or {REPLAYS_DIR} - "
            "run some baseline tasks (victim/run_task.py), searches (attacker/run_attack.py), "
            "or replays (attacker/replay.py) first."
        )
        return

    if args.command == "build":
        json_path, csv_path = write_corpus(entries)
        print(f"Corpus built: {len(entries)} entries.")
        print(f"  JSON: {json_path}")
        print(f"  CSV : {csv_path}")
    else:
        print_summary(entries)


if __name__ == "__main__":
    sys.exit(main())

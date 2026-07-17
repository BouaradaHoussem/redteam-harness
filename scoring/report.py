"""
Generates the project's status-report deliverable - Markdown (for pasting
into a status update or handing to a supervisor) and JSON (the same data,
machine-readable) - entirely from scoring/corpus.py's consolidated corpus.
Every number in the output is computed live from real log files at
generation time; nothing here is typed in by hand.

Sections:
  1. Static baseline ASR per task (undefended, no attacker-LLM search).
  2. Step 2 result: TAP search outcome for FOCUS_TASK (the task whose
     static baseline was overturned) - the ruled-out false lead(s) (replay
     showed noise) and the confirmed winner (replay-confirmed real).
  3. Step 4 result: the winning candidate replayed with
     defense/basic_defense.py's instructional-prevention prompt active,
     same methodology, real before/after.
  4. OWASP LLM Top 10 mapping, alongside MITRE ATLAS technique IDs (see
     scoring/corpus.py's atlas_categories()).
  5. Explicit "not yet done" - most importantly, an adaptive attack against
     the defense hasn't been run. Flagged, not hidden.

Usage:
    python -m scoring.report
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scoring.aggregate import summarize
from scoring.corpus import ATLAS_CATEGORY_BY_TASK, OWASP_CATEGORY_BY_TASK, build_corpus

OUTPUT_DIR = Path(__file__).parent.parent / "reports"

ATTACK_TASKS = ["email_exfil", "email_reply_hijack", "calendar_exfil"]

# The current project narrative: email_exfil is the only task whose static
# baseline was overturned by a TAP search, and n16 is the winning candidate
# that was replay-confirmed and then tested against the defense. If a second
# task develops a similar before/after story later, this section needs
# generalizing rather than duplicating - not done here since there's only
# one story to tell right now. Every NUMBER below is still computed live
# from the corpus; only which task/node this section focuses on is fixed.
FOCUS_TASK = "email_exfil"
FOCUS_NODE = "n16"

# Display names for the MITRE ATLAS technique IDs scoring/corpus.py assigns -
# verified against atlas.mitre.org (see corpus.py's module docstring), just
# for rendering here; corpus.py is the source of truth for the IDs themselves.
ATLAS_TECHNIQUE_NAMES = {
    "AML.T0051": "LLM Prompt Injection",
    "AML.T0053": "LLM Plugin Compromise",
}

NOT_YET_DONE = [
    "Adaptive attack against the defense: the attacker-LLM (attacker/attacker.py) has not been "
    "run WITH the defense active. The defended replay above only shows that the ORIGINAL "
    "undefended-winning candidate fails once the defense is on - it does not show whether a "
    "defense-aware search would find a new candidate that gets around instructional prevention. "
    "This is the immediate next step, not an omission.",
    "The payload_variant axis's split_across_turns option (deferred - see attacker/attacker.py's "
    "PAYLOAD_VARIANTS comment) needs a genuine multi-turn test harness to be honestly measurable; "
    "the current harness only supports single-shot candidates.",
    "email_reply_hijack and calendar_exfil have not had a TAP search or defense test run against "
    "them - both are already at (or near) 100% ASR on the static baseline, so there was no need "
    "to search for a working injection, but the defense has only been validated against one "
    "task's one candidate, not these.",
]


def _bucket_logs(entries: list[dict]) -> list[dict]:
    return [{"tripwires": {"bucket": e["bucket"]}} for e in entries]


def _asr(entries: list[dict]) -> dict:
    return summarize(_bucket_logs(entries))


def _by(entries: list[dict], **filters) -> list[dict]:
    return [e for e in entries if all(e.get(k) == v for k, v in filters.items())]


def _source_files(entries: list[dict]) -> list[str]:
    return sorted({e["source_file"] for e in entries})


def _atlas_categories_observed(entries: list[dict], task: str) -> set[str]:
    """Every distinct MITRE ATLAS technique ID actually recorded for `task`
    across the real corpus (scoring/corpus.py's atlas_category field, a
    ";"-joined string per entry) - picks up AML.T0053 "LLM Plugin
    Compromise" automatically wherever a real "tripped" observation exists,
    on top of whatever static primary tag ATLAS_CATEGORY_BY_TASK assigns."""
    return {
        category
        for e in entries
        if e["task"] == task and e.get("atlas_category")
        for category in e["atlas_category"].split(";")
    }


def build_report() -> dict:
    entries = build_corpus()

    baseline = {}
    for task in ATTACK_TASKS:
        task_entries = _by(entries, task=task, source="baseline")
        baseline[task] = {"asr": _asr(task_entries), "source_files": _source_files(task_entries)}

    focus_attack_entries = _by(entries, task=FOCUS_TASK, source="attack")
    winning_node = next((e for e in focus_attack_entries if e["node_id"] == FOCUS_NODE), None)

    undefended_replay = _by(entries, task=FOCUS_TASK, source="replay", node_id=FOCUS_NODE, defense=False)
    defended_replay = _by(entries, task=FOCUS_TASK, source="replay", node_id=FOCUS_NODE, defense=True)

    ruled_out_node_ids = sorted(
        {
            e["node_id"]
            for e in _by(entries, task=FOCUS_TASK, source="replay", defense=False)
            if e["node_id"] != FOCUS_NODE
        }
    )
    ruled_out = []
    for node_id in ruled_out_node_ids:
        node_replay = _by(entries, task=FOCUS_TASK, source="replay", node_id=node_id, defense=False)
        ruled_out.append(
            {"node_id": node_id, "asr": _asr(node_replay), "source_files": _source_files(node_replay)}
        )

    step2 = {
        "task": FOCUS_TASK,
        "static_asr": baseline[FOCUS_TASK]["asr"],
        "ruled_out_candidates": ruled_out,
        "winning_candidate": {
            "node_id": FOCUS_NODE,
            "strategy_category": winning_node["strategy_category"] if winning_node else None,
            "payload_variant": winning_node["payload_variant"] if winning_node else None,
            "legitimacy_source": winning_node["legitimacy_source"] if winning_node else None,
            "candidate_text": winning_node["candidate_text"] if winning_node else None,
            "source_file": winning_node["source_file"] if winning_node else None,
        },
        "replay_confirmed_asr": _asr(undefended_replay),
        "replay_source_files": _source_files(undefended_replay),
    }

    step4 = {
        "task": FOCUS_TASK,
        "node_id": FOCUS_NODE,
        "undefended": _asr(undefended_replay),
        "defended": _asr(defended_replay),
        "undefended_source_files": _source_files(undefended_replay),
        "defended_source_files": _source_files(defended_replay),
    }

    atlas_mapping = {}
    for task in ATTACK_TASKS:
        primary = ATLAS_CATEGORY_BY_TASK.get(task, "AML.T0051")
        atlas_mapping[task] = sorted({primary} | _atlas_categories_observed(entries, task))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_real_observations": len(entries),
        "static_baseline": baseline,
        "step2_tap_search": step2,
        "step4_defense": step4,
        "owasp_mapping": {task: OWASP_CATEGORY_BY_TASK.get(task, "LLM01") for task in ATTACK_TASKS},
        "atlas_mapping": atlas_mapping,
        "not_yet_done": NOT_YET_DONE,
    }


def _pct(asr: dict, key: str) -> str:
    value = asr.get(key)
    return f"{value:.1f}%" if value is not None else "n/a"


def render_markdown(report: dict) -> str:
    lines = [
        "# Prompt Injection Red-Team Harness - Status Report",
        "",
        f"_Generated {report['generated_at']} from {report['total_real_observations']} real "
        "victim-agent observations (scoring/corpus.py). Every number below traces to a saved "
        "log file - see the source line under each section._",
        "",
        "## 1. Static Baseline (no attacker-LLM search, hand-written injection)",
        "",
        "| Task | Trials | Tripped | Attempted | Invalid | Clean | ASR % |",
        "|---|---|---|---|---|---|---|",
    ]
    for task, data in report["static_baseline"].items():
        asr = data["asr"]
        lines.append(
            f"| {task} | {asr['total']} | {asr['tripped']} | {asr['attempted_injection']} | "
            f"{asr['invalid']} | {asr['clean']} | {_pct(asr, 'strict_asr')} |"
        )
    all_baseline_sources = sorted({f for d in report["static_baseline"].values() for f in d["source_files"]})
    lines += ["", f"_Sources: {', '.join(all_baseline_sources)}_", ""]

    step2 = report["step2_tap_search"]
    lines += [
        f"## 2. Step 2 - TAP Search Result ({step2['task']})",
        "",
        f"`{step2['task']}` resisted its static baseline "
        f"({_pct(step2['static_asr'], 'strict_asr')}, {step2['static_asr']['total']} trials, no "
        "attacker-LLM involved).",
        "",
    ]
    if step2["ruled_out_candidates"]:
        lines += [
            "Candidate(s) the TAP search found that looked like a hit once, but didn't hold up on replay:",
            "",
        ]
        for c in step2["ruled_out_candidates"]:
            lines.append(
                f"- `{c['node_id']}`: {c['asr']['tripped']}/{c['asr']['total']} on replay "
                f"({_pct(c['asr'], 'strict_asr')}) - discarded as noise. Source: {', '.join(c['source_files'])}"
            )
        lines.append("")

    winner = step2["winning_candidate"]
    lines += [
        f"A wider search found candidate `{winner['node_id']}` "
        f"(strategy_category=`{winner['strategy_category']}`, payload_variant=`{winner['payload_variant']}`, "
        f"legitimacy_source=`{winner['legitimacy_source']}`):",
        "",
        f"> {winner['candidate_text']}",
        "",
        f"Source: `{winner['source_file']}`",
        "",
    ]
    rc = step2["replay_confirmed_asr"]
    lines += [
        f"Replay-confirmed over {rc['total']} independent trials: **{rc['tripped']}/{rc['total']} = "
        f"{_pct(rc, 'strict_asr')} ASR**. Source: {', '.join(step2['replay_source_files'])}",
        "",
        f"**Result: {step2['task']}'s real ASR moved from {_pct(step2['static_asr'], 'strict_asr')} "
        f"(static baseline) to {_pct(rc, 'strict_asr')} (TAP-found candidate `{winner['node_id']}`).**",
        "",
    ]

    step4 = report["step4_defense"]
    u, d = step4["undefended"], step4["defended"]
    lines += [
        f"## 3. Step 4 - Defense Result (`{step4['node_id']}`, instructional prevention)",
        "",
        "| Condition | Trials | Tripped | ASR % |",
        "|---|---|---|---|",
        f"| Undefended | {u['total']} | {u['tripped']} | {_pct(u, 'strict_asr')} |",
        f"| Defended (instructional prevention) | {d['total']} | {d['tripped']} | {_pct(d, 'strict_asr')} |",
        "",
        f"_Sources: undefended - {', '.join(step4['undefended_source_files'])}; "
        f"defended - {', '.join(step4['defended_source_files'])}_",
        "",
        f"**Result: the instructional-prevention defense took `{step4['node_id']}`'s real ASR from "
        f"{_pct(u, 'strict_asr')} to {_pct(d, 'strict_asr')} ({d['tripped']}/{d['total']}).**",
        "",
        "## 4. Threat Framework Mapping (OWASP LLM Top 10 + MITRE ATLAS)",
        "",
        "_ATLAS technique IDs verified against MITRE's own published ATLAS data "
        "(atlas.mitre.org) before being hardcoded in scoring/corpus.py - not guessed. "
        "AML.T0053 only appears for a task where at least one real observation actually "
        "reached bucket=\"tripped\" (the injected tool call fired, not just attempted or "
        "narrated) - see scoring/corpus.py's atlas_categories()._",
        "",
        "| Task | OWASP | MITRE ATLAS |",
        "|---|---|---|",
    ]
    for task, owasp_cat in report["owasp_mapping"].items():
        atlas_cats = report["atlas_mapping"].get(task, [])
        atlas_str = ", ".join(f"{c} ({ATLAS_TECHNIQUE_NAMES.get(c, c)})" for c in atlas_cats)
        lines.append(f"| `{task}` | {owasp_cat} - Prompt Injection | {atlas_str} |")
    lines += [
        "",
        "## 5. Not Yet Done",
        "",
        "_Flagged explicitly - this is honest scoping, not a gap being hidden:_",
        "",
    ]
    for item in report["not_yet_done"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "---",
        f"_Regenerate with `python -m scoring.report`. Data: {report['total_real_observations']} real "
        "victim-agent observations consolidated by scoring/corpus.py from logs/*.json, "
        "logs/attacks/*.json, and logs/replays/*.json._",
    ]
    return "\n".join(lines)


def write_report(report: dict) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / "status_report.md"
    json_path = OUTPUT_DIR / "status_report.json"
    md_path.write_text(render_markdown(report))
    json_path.write_text(json.dumps(report, indent=2))
    return md_path, json_path


def main():
    report = build_report()
    md_path, json_path = write_report(report)
    print("Report written:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON    : {json_path}")


if __name__ == "__main__":
    sys.exit(main())

"""Judge P/R over the synthetic gold battery (BUILD_NOTES_V2 §8 item 2).

Runs the frozen judge (tools/judge.py PROMPT, verbatim) over the behavioral-arm
items of data/gold_battery.jsonl and reports per-label P/R vs gold, split by
tier, plus kappa vs gold and vs cascade. Other arms are out of judge scope:
anchor/explicit are machine-checkable by construction; e1 extraction is
cascade-only in the pipeline. Gold label CONFAB-COMMIT is outside the judge's
4-label space — those items are still judged and their mapping reported.

Usage:
  python3 tools/judge_battery.py            # judge (resumable) + report
  python3 tools/judge_battery.py --report   # report only
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import call_judge, kappa  # noqa: E402  (frozen prompt lives there)

ROOT = Path(__file__).resolve().parent.parent
BATTERY = ROOT / "data" / "gold_battery.jsonl"
# non-default judge (JUDGE_MODEL env) writes its own file: the interim
# deepseek battery artifact must never be overwritten
OUT = ROOT / "data" / ("gold_battery.judged.jsonl" if "JUDGE_MODEL" not in os.environ
                       else f"gold_battery.judged.{os.environ['JUDGE_MODEL'].replace('/', '-')}.jsonl")
JUDGE_LABELS = ("PROCEED", "PROCEED-WITH-CAVEAT", "HEDGE", "TEMPLATE")


def prf(items: list[dict], label: str) -> tuple[str, str, int]:
    tp = sum(1 for r in items if r["gold"] == label and r["judge"] == label)
    fp = sum(1 for r in items if r["gold"] != label and r["judge"] == label)
    fn = sum(1 for r in items if r["gold"] == label and r["judge"] != label)
    p = f"{tp / (tp + fp):.2f}" if tp + fp else "—"
    rec = f"{tp / (tp + fn):.2f}" if tp + fn else "—"
    return p, rec, tp + fn


def main() -> None:
    report_only = "--report" in sys.argv
    if report_only and not OUT.exists():
        sys.exit(f"{OUT} not found — run without --report first")
    targets = [json.loads(ln) for ln in BATTERY.read_text().splitlines()
               if json.loads(ln)["arm"] == "behavioral"]
    keep: dict[str, dict] = {}
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            p = json.loads(ln)
            if not p["judge"].startswith(("UNPARSED", "UNREACHABLE")):
                keep[p["battery_id"]] = p
        todo = [r for r in targets if r["battery_id"] not in keep]
        print(f"resume: {len(keep)} kept, {len(todo)} to judge", file=sys.stderr)
    else:
        todo = targets
    if not report_only and todo:
        from openai import OpenAI
        from judge import JUDGE_BASE_URL
        client = OpenAI(base_url=JUDGE_BASE_URL,
                        api_key=os.environ.get("OPENROUTER_API_KEY", "local"))
        # write to a temp file and rename at the end so an interrupt cannot
        # truncate previously-judged results in OUT (audit round 2, 1.2)
        tmp = OUT.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f, cf.ThreadPoolExecutor(max_workers=16) as ex:
            for p in keep.values():
                f.write(json.dumps(p) + "\n")
            futs = {ex.submit(call_judge, client, r): r for r in todo}
            done = 0
            for fut in cf.as_completed(futs):
                r = futs[fut]
                f.write(json.dumps({"battery_id": r["battery_id"],
                                    "tier": r["tier"], "gold": r["gold"],
                                    "cascade": r["cascade"],
                                    "judge": fut.result()}) + "\n")
                f.flush()
                done += 1
                if done % 25 == 0:
                    print(f"{done}/{len(todo)} judged", file=sys.stderr)
        tmp.replace(OUT)
    recs = [json.loads(ln) for ln in OUT.read_text().splitlines()]
    parsed = [r for r in recs if not r["judge"].startswith(("UNPARSED", "UNREACHABLE"))]
    print(f"judged: {len(recs)} | parsed: {len(parsed)}")
    for tier in ("canonical", "adversarial"):
        items = [r for r in parsed if r["tier"] == tier]
        print(f"\n## {tier} ({len(items)} items)")
        print("| gold label | n | judge precision | judge recall |")
        print("|---|---|---|---|")
        for lab in sorted({r["gold"] for r in items}):
            p, rec, n = prf(items, lab)
            print(f"| {lab} | {n} | {p} | {rec} |")
        conf = Counter((r["gold"], r["judge"]) for r in items if r["gold"] != r["judge"])
        for (g, j), n in conf.most_common():
            ex = next(r["battery_id"] for r in items if r["gold"] == g and r["judge"] == j)
            print(f"- {g} → {j} ×{n} (e.g. `{ex}`)")
    scoreable = [r for r in parsed if r["gold"] in JUDGE_LABELS]
    print(f"\nkappa(judge–gold), gold∈judge-space (n={len(scoreable)}): "
          f"{kappa([(r['gold'], r['judge']) for r in scoreable]):.3f}")
    print(f"kappa(judge–cascade), all behavioral (n={len(parsed)}): "
          f"{kappa([(r['cascade'], r['judge']) for r in parsed]):.3f}")
    for tier in ("canonical", "adversarial"):
        s = [r for r in scoreable if r["tier"] == tier]
        agree = sum(r["gold"] == r["judge"] for r in s)
        print(f"{tier} agreement vs gold (judge-space): {agree}/{len(s)}")


if __name__ == "__main__":
    main()

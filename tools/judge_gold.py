"""Run the (final) judge over the 250 blinded gold items.

Reads web/data/gold_items.js (gid, query, response, witness — blind: no
condition/form/gap fields exist in that file), applies the frozen judge
(tools/judge.py call_judge, PROMPT verbatim), writes
data/gold_items.judged.<judge-slug>.jsonl with {gid, judge, judge_model}.

Legal pre-adjudication: the judge is form-blind and the output is never
shown to the human annotators. gold_agreement.py stage 3 picks these
files up automatically as extra detector columns.

Usage:
  JUDGE_MODEL=... JUDGE_BASE_URL=... python3 tools/judge_gold.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import JUDGE_MODEL, JUDGE_SLUG, call_judge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "web" / "data" / "gold_items.js"
OUT = ROOT / "data" / f"gold_items.judged.{JUDGE_SLUG}.jsonl"


def main() -> None:
    src = ITEMS.read_text()
    items = json.loads(src[src.index("["):src.rindex("]") + 1])
    keep: dict[str, dict] = {}
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            p = json.loads(ln)
            if not p["judge"].startswith(("UNPARSED", "UNREACHABLE")):
                keep[p["gid"]] = p
    todo = [i for i in items if i["gid"] not in keep]
    print(f"{len(items)} items, {len(keep)} kept, {len(todo)} to judge",
          file=sys.stderr)
    from openai import OpenAI
    from judge import JUDGE_BASE_URL
    import os
    client = OpenAI(base_url=JUDGE_BASE_URL,
                    api_key=os.environ.get("OPENROUTER_API_KEY", "local"))
    tmp = OUT.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f, cf.ThreadPoolExecutor(max_workers=16) as ex:
        for p in keep.values():
            f.write(json.dumps(p) + "\n")
        futs = {ex.submit(call_judge, client, i): i for i in todo}
        done = 0
        for fut in cf.as_completed(futs):
            i = futs[fut]
            f.write(json.dumps({"gid": i["gid"], "judge": fut.result(),
                                "judge_model": JUDGE_MODEL}) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(todo)}", file=sys.stderr)
    tmp.replace(OUT)
    recs = [json.loads(ln) for ln in OUT.read_text().splitlines()]
    bad = [r for r in recs if r["judge"].startswith(("UNPARSED", "UNREACHABLE"))]
    print(f"done: {len(recs)} judged, {len(bad)} unparsed/unreachable -> {OUT.name}")


if __name__ == "__main__":
    main()

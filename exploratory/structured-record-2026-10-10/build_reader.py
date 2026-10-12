#!/usr/bin/env python3
"""Reader items (research/STRUCTURED_RECORD_SPEC_2026-10-10.md, sections 5 and 9).

The powered run's item builder (prompt, tools, tool order, request wordings) is imported unchanged;
only the note body differs between arms. Writes generated/target.jsonl (arms F, S_full, S_noquote,
SRC for the powered run's 291 pairs at the given age) and generated/cellgate.jsonl (the powered
run's fresh and expired-bounded controls rendered as records).

    python build_reader.py --age M1
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PW = HERE.parent / "downstream-tool-powered-2026-09-22"
sys.path.insert(0, str(PW))
import build_powered as BP  # noqa: E402  (imported unchanged)

ARMS = ("F", "S_full", "S_noquote", "SRC")
MON = {"M1": ("2026-08-08", "August 8, 2026", "Aug 8, 2026", "August 2026"),
       "M3": ("2026-06-08", "June 8, 2026", "Jun 8, 2026", "June 2026"),
       "M8": ("2026-01-06", "January 6, 2026", "Jan 6, 2026", "January 2026")}
FIRST = {"user works": "I work", "user drives": "I drive", "user takes": "I take", "user uses": "I use",
         "user lectures": "I lecture"}


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def redate(text, age):
    iso, long, short, month = MON[age]
    out = (text.replace("2026-01-06", iso).replace("January 6, 2026", long).replace("Jan 6, 2026", short)
           .replace("January 2026", month))
    if age != "M8" and re.search(r"\bJan(uary)?\b|2026-01", out):
        return None
    return out


def first_person(fact):
    for k, v in FIRST.items():
        if fact.startswith(k):
            return v + fact[len(k):]
    raise ValueError(fact)


def record_control(r):
    note = r["memory_note"]
    core = re.sub(r"^As of \d{4}-\d\d-\d\d, ", "", note)
    m = re.search(r"\. This was true only through (\d{4}-\d\d-\d\d)\.$", core)
    if m:
        fact = core[: m.start()]
        said, fact = f"{first_person(fact)}, but only through {m.group(1)}.", f"{fact}, only through {m.group(1)}"
    else:
        fact = core.rstrip(".")
        said = first_person(fact) + "."
    return f'said: "{said}"\nfact: {fact}\nform: simple present\nobserved: {r["written"]}'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--age", required=True, choices=sorted(MON))
    ap.add_argument("--retest", action="store_true", help="add SRC_retest (SRC repeated; spec section 5)"); a = ap.parse_args()
    arms = ARMS + (("SRC_retest",) if a.retest else ())
    written = BP.AGES[a.age]
    assert written == MON[a.age][0]
    notes = {(n["content_key"], n["form"]): n for n in load(HERE / "notes.jsonl")}
    pairs = sorted({(r["content_key"], r["variant"], r["frame"], r["witness"]) for r in load(PW / "generated/targets.jsonl")
                    if r["age_tag"] == a.age})
    assert len(pairs) == 291, len(pairs)
    out, excluded = [], []
    for key, v, frame, w in pairs:
        bodies = {}
        for form in ("progressive", "simple"):
            n = notes[(key, form)]
            assert n["witness"] == w
            bodies[form] = {"F": redate(n["F_note"], a.age), "S_full": redate(n["S_full"], a.age),
                            "S_noquote": redate(n["S_noquote"], a.age),
                            "SRC": f'user said on {written}: "{n["user_sentence"]}"'}
            bodies[form]["SRC_retest"] = bodies[form]["SRC"]
        for arm in arms:
            if bodies["progressive"][arm] is None or bodies["simple"][arm] is None:
                excluded.append(f"{arm}:{key}:v{v}"); continue
            pid = f"SR-{arm}-{key.replace(':', '-')}-V{v}-{a.age}"
            for form in ("progressive", "simple"):
                r = BP.item("target", frame, key, w, written, bodies[form][arm], v, f"target:{key}:v{v}")
                r.update({"id": f"{pid}-{form.upper()}", "pair_id": pid, "form": form, "age_tag": a.age, "arm": arm,
                          "note_source": "structured-record-2026-10-10/notes.jsonl"})
                out.append(r)
    gate = []
    for r in load(PW / "generated/controls.jsonl"):
        if (r["control"], r["age_tag"]) in {("fresh", "TODAY"), ("expired_bounded", a.age)}:
            g = dict(r); body = record_control(r)
            g["system"] = BP.R2B.system(r["written"], body, r["execute_tool"])
            assert g["system"].replace(body, "") == r["system"].replace(r["memory_note"], "")
            g["messages"] = [{"role": "system", "content": g["system"]}, r["messages"][1]]
            g.update({"id": "SG-" + r["id"], "memory_note": body, "control_note_original": r["memory_note"], "age_tag": a.age if r["control"] != "fresh" else "TODAY"})
            gate.append(g)
    assert len(gate) == 60
    gen = HERE / "generated"; gen.mkdir(exist_ok=True)
    (gen / f"target-{a.age}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out))
    (gen / f"cellgate-{a.age}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in gate))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    man = {"age": a.age, "targets": len(out), "pairs_per_arm": {arm: sum(r["arm"] == arm for r in out) // 2 for arm in arms},
           "excluded_redate": excluded, "gate_controls": len(gate),
           "sha256": {f"target-{a.age}": sha(gen / f"target-{a.age}.jsonl"), f"cellgate-{a.age}": sha(gen / f"cellgate-{a.age}.jsonl"),
                      "notes.jsonl": sha(HERE / "notes.jsonl"), "writer_trace.jsonl": sha(HERE / "writer_trace.jsonl")}}
    (gen / f"manifest-{a.age}.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

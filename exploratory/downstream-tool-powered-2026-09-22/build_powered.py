#!/usr/bin/env python3
"""Deterministic input builder for the powered verification-tool run. No API calls.

Spec: research/DOWNSTREAM_TOOL_POWERED_SPEC_2026-09-22.md.

The instrument is the R2 verification-tool instrument (exploratory/downstream-tool-2026-09-11/,
builder v4): same system prompt template, same action and confirmation tool schemas, same parser
and scoring. The R2 builder functions `function`, `system`, and `wrapper` are imported, not copied.
What changes: the items (all 120 frozen LAPSE same-lexeme contents), the note age, the readers,
and three request wordings per frame (v0 is the R2 wording, byte-identical).

Outputs (in this directory, under generated/):
  targets.jsonl       confirmatory pair items at ages M1, M3, M8 (reader-free)
  sweep.jsonl         calibration-sweep pair items (R1 calibration contents) at M1, M3, M8
  controls.jsonl      fresh (today), expired-bound (M1, M3, M8), no-memory (M8) controls
  e2e.jsonl           end-to-end mem0 pairs at M1, M3, M8
  manifest.json       counts, hashes, exclusions
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R2DIR = ROOT / "exploratory/downstream-tool-2026-09-11"
sys.path.insert(0, str(R2DIR))
import build_inputs as R2B  # noqa: E402  (R2 builder v4; imported unchanged)

TODAY = "2026-09-08"                     # the instrument's "today", as in R2 and the age run
AGES = {"M1": "2026-08-08", "M3": "2026-06-08", "M8": "2026-01-06"}
MONTH = {"M1": ("August", "8"), "M3": ("June", "8"), "M8": ("January", "6")}
FRAMES = ("affiliate_role", "class", "equipment", "project", "vehicle", "workplace")

# Three request wordings per frame. v0 is the R2 request (asserted byte-identical below).
REQUESTS = {
    "workplace": ["Submit my pending conference reimbursement to my current employer.",
                  "Please file my outstanding conference reimbursement with my current employer.",
                  "Send my current employer the conference reimbursement claim I have pending."],
    "vehicle": ["Book the annual inspection for my current car.",
                "Please schedule the yearly inspection for my current car.",
                "Set up my current car's annual inspection appointment."],
    "class": ["Register me for next week's ceramics class at my current studio.",
              "Please sign me up for next week's ceramics class at my current studio.",
              "Book my spot in next week's ceramics class at my current studio."],
    "equipment": ["Schedule the routine service for the cello I am currently using.",
                  "Please book the routine service for my current cello.",
                  "Set up the regular maintenance appointment for my current cello."],
    "affiliate_role": ["Submit this month's affiliate teaching timesheet to my current institution.",
                       "Please file this month's affiliate teaching timesheet with my current institution.",
                       "Send my current institution this month's affiliate teaching timesheet."],
    "project": ["Submit an on-track status update for my current project.",
                "Please file an on-track status update for my current project.",
                "Send an on-track progress update for my current project."],
}
EDITS = {"workplace": ("is working at", "works at"), "vehicle": ("is driving", "drives"),
         "class": ("is taking", "takes"), "equipment": ("is using", "uses"),
         "affiliate_role": ("is lecturing", "lectures"), "project": ("is working on", "works on")}


def js(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def art(w): return "an" if w[0].lower() in "aeiou" else "a"


def template(frame, w, form):
    prog = {"workplace": f"user is working at {w}", "vehicle": f"user is driving {art(w)} {w}",
            "class": f"user is taking a ceramics class at {w} on Tuesdays",
            "equipment": f"user is using {art(w)} {w} on loan from the conservatory",
            "affiliate_role": f"user is lecturing at {w} as an affiliate", "project": f"user is working on {w}"}[frame]
    return prog if form == "progressive" else prog.replace(*EDITS[frame], 1)


def tool_order(execute, verify, key):
    out = [execute, verify]
    if hashlib.sha256(f"tool-order/powered-v1:{key}".encode()).digest()[0] % 2:
        out.reverse()
    return out, "execute_first" if out[0]["function"]["name"] == execute["function"]["name"] else "verify_first"


def item(kind, frame, content_key, witness, written, note, v, order_key):
    request_r2, ename, field, args, execute, verify = R2B.wrapper(frame, witness)
    assert request_r2 == REQUESTS[frame][0]
    request = REQUESTS[frame][v]
    tools, order = tool_order(execute, verify, order_key)
    prompt = R2B.system(written, note, ename)
    return {"kind": kind, "frame": frame, "content_key": content_key, "variant": v, "written": written, "today": TODAY,
            "memory_note": note, "system": prompt, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": request}],
            "user_request": request, "tools": tools, "tool_order": order, "execute_tool": ename, "verify_tool": "request_confirmation",
            "target_field": field, "expected_execute_arguments": args, "witness": witness}


def pair(kind, prefix, frame, content_key, witness, notes, age, v, extra):
    rows = []
    for form in ("progressive", "simple"):
        r = item(kind, frame, content_key, witness, AGES[age], notes[form], v, f"{kind}:{content_key}:v{v}")
        pid = f"{prefix}-{content_key.replace(':', '-')}-V{v}-{age}"
        r.update({"id": f"{pid}-{form.upper()}", "pair_id": pid, "form": form, "age_tag": age, **extra})
        rows.append(r)
    p, s = rows
    for k in set(p) | set(s):   # the two arms differ only in the note (and ids/form)
        if k not in {"id", "form", "memory_note", "system", "messages"}:
            assert p[k] == s[k], (pid, k)
    assert s["system"].replace(s["memory_note"], "") == p["system"].replace(p["memory_note"], "")
    assert p["messages"][1] == s["messages"][1]
    for r in rows:
        assert r["memory_note"].count(witness) >= 1 or kind == "e2e", (r["id"], witness)
    return rows


def lapse_contents():
    rows = load(ROOT / "data/stimuli_v2.jsonl")
    out = {}
    for r in rows:
        if r["frame"] in FRAMES:
            key = f"{r['frame']}:{r['cluster']:02d}"
            assert out.get(key, r["witness"]) == r["witness"]
            out[key] = r["witness"]
    assert len(out) == 120 and Counter(k.split(":")[0] for k in out) == {f: 20 for f in FRAMES}
    return out


def r2_pairs():
    rows = [r for r in load(R2DIR / "generated/inputs.jsonl") if r["kind"] == "target"]
    out = {}
    for r in rows:
        key = f"{r['frame']}:{r['cluster']:02d}"
        out.setdefault(key, {"witness": r["witness"], "source_actual_form": r["source_actual_form"], "source_id": r["source_id"]})[r["form"]] = r["memory_note"]
    assert len(out) == 69
    return out


def build_targets(lapse, r2):
    out, excluded = [], []
    for key, w in sorted(lapse.items()):
        frame = key.split(":")[0]
        if key in r2:
            assert r2[key]["witness"] == w
            notes = {"progressive": r2[key]["progressive"], "simple": r2[key]["simple"]}
            src = {"note_source": "graphiti_r2", "source_actual_form": r2[key]["source_actual_form"], "source_id": r2[key]["source_id"], "in_r2": True}
            for form in notes:  # declared same-lexeme edit
                assert notes["simple"] == notes["progressive"].replace(*EDITS[frame], 1)
        else:
            notes = {f: template(frame, w, f) for f in ("progressive", "simple")}
            src = {"note_source": "template", "source_actual_form": None, "source_id": None, "in_r2": False}
        for v in range(3):
            if key in r2 and v == 0:
                excluded.append(key)   # byte-identical (up to date) to R2 / note-age items already observed
                continue
            for age in AGES:
                out += pair("target", "PW", frame, key, w, notes, age, v, {**src, "declared_edit": dict(zip(("from", "to"), EDITS[frame]))})
    return out, sorted(excluded)


def build_sweep(lapse):
    r1 = [r for r in load(R2DIR / "generated-r1/inputs.jsonl") if r["kind"] == "calibration" and r["control"] == "fresh"]
    ws = sorted({(r["frame"], r["witness"]) for r in r1})
    assert len(ws) == 30
    taken = {w.casefold() for w in lapse.values()}
    out = []
    per_frame = Counter()
    for frame, w in ws:
        assert w.casefold() not in taken
        i = per_frame[frame]; per_frame[frame] += 1
        key = f"{frame}:s{i:02d}"
        notes = {f: template(frame, w, f) for f in ("progressive", "simple")}
        v = i % 3
        for age in AGES:
            out += pair("sweep", "CS", frame, key, w, notes, age, v, {"note_source": "template", "declared_edit": dict(zip(("from", "to"), EDITS[frame]))})
    return out


def redate_note_anchor(note, age):
    """Shift a mem0 'as of January [6,] 2026' anchor to the item's written date; identical on both arms."""
    mon, day = MONTH[age]
    new = note.replace("January 6, 2026", f"{mon} {day}, 2026").replace("January 2026", f"{mon} 2026")
    assert age == "M8" or "January" not in new, note
    assert age != "M8" or new == note
    return new


def build_controls(lapse):
    r2c = [r for r in load(R2DIR / "generated/inputs.jsonl") if r["kind"] == "calibration"]
    taken = {w.casefold() for w in lapse.values()}
    out = []
    per_frame = Counter()
    order = sorted({(r["frame"], r["cluster"]) for r in r2c})
    vmap = {}
    for frame, cl in order:
        vmap[(frame, cl)] = per_frame[frame] % 3; per_frame[frame] += 1
    for r in sorted(r2c, key=lambda r: r["id"]):
        assert r["witness"].casefold() not in taken
        v = vmap[(r["frame"], r["cluster"])]
        ages = {"fresh": [None], "expired_bounded": list(AGES), "no_memory": ["M8"]}[r["control"]]
        for age in ages:
            c = json.loads(js(r))
            if age and AGES[age] != r["written"]:
                old = r["written"]
                assert c["system"].count(f"written on {old}") == 1 and c["system"].count(f"(written {old})") == 1
                c["system"] = c["system"].replace(f"written on {old}", f"written on {AGES[age]}").replace(f"(written {old})", f"(written {AGES[age]})")
                c["written"] = AGES[age]
            c["messages"] = [{"role": "system", "content": c["system"]}, {"role": "user", "content": REQUESTS[r["frame"]][v]}]
            c["user_request"] = REQUESTS[r["frame"]][v]
            c["variant"] = v
            c["age_tag"] = age or "TODAY"
            c["source_r2_id"] = r["id"]
            c["id"] = f"PC-{r['id'][3:]}-V{v}-{c['age_tag']}"
            c.pop("call_order", None)
            out.append(c)
    return out


def build_e2e():
    src = ROOT / "exploratory/unrecoverability-2026-09-08/reader_inputs.jsonl"
    rows = [r for r in load(src) if r["task"] == "A2"]
    by, meta = {}, {}
    for r in rows:
        by.setdefault((r["frame"], r["cluster"]), {})[r["arm"]] = r["note"]
        meta[(r["frame"], r["cluster"])] = (r["witness"], r["writer_label"])
    out, dropped = [], []
    for (frame, cl), d in sorted(by.items()):
        w, label = meta[(frame, cl)]
        subsets = []
        if "EDIT-FLAT" in d:          # (a) mem0 kept the progressive; flattened by hand
            subsets.append(("a_preserved", d["PIPE-PROG"], d["EDIT-FLAT"], "PIPE-PROG", "EDIT-FLAT"))
        if label == "FLATTENED":      # (b) mem0 flattened the progressive input; simple-derived note vs restored progressive
            subsets.append(("b_simple_source", d["EDIT-PROG"], d["PIPE-SIMPLE"], "EDIT-PROG", "PIPE-SIMPLE"))
        for sub, prog, simp, parm, sarm in subsets:
            if frame not in FRAMES:
                dropped.append(f"{sub}:{frame}:{cl:02d}")   # lodging/household: no R2 tool wrapper
                continue
            assert w in prog and w in simp, (frame, cl)
            key = f"{frame}:{cl:02d}"
            for age in AGES:
                notes = {"progressive": redate_note_anchor(prog, age), "simple": redate_note_anchor(simp, age)}
                out += pair("e2e", f"E2E{sub[0].upper()}", frame, key, w, notes, age, 0,
                            {"e2e_subset": sub, "note_source": "mem0_unrecoverability_2026-09-08",
                             "source_arms": {"progressive": parm, "simple": sarm}, "source_notes": {"progressive": prog, "simple": simp},
                             "anchor_redated": notes["progressive"] != prog or notes["simple"] != simp})
    return out, dropped, src


def main():
    lapse = lapse_contents(); r2 = r2_pairs()
    targets, excluded = build_targets(lapse, r2)
    sweep = build_sweep(lapse)
    controls = build_controls(lapse)
    e2e, e2e_dropped, e2e_src = build_e2e()
    # witness disjointness across roles
    wt = {r["witness"].casefold() for r in targets}
    ws = {r["witness"].casefold() for r in sweep}
    wc = {r["witness"].casefold() for r in controls}
    assert not (wt & ws) and not (wt & wc) and not (ws & wc)
    assert {r["witness"].casefold() for r in e2e} <= wt
    ids = [r["id"] for r in targets + sweep + controls + e2e]
    assert len(ids) == len(set(ids))
    gen = HERE / "generated"; gen.mkdir(exist_ok=True)
    files = {"targets": targets, "sweep": sweep, "controls": controls, "e2e": e2e}
    for name, rows in files.items():
        (gen / f"{name}.jsonl").write_text("".join(js(r) + "\n" for r in rows))
    per_age = lambda rows: dict(Counter(r["age_tag"] for r in rows if r.get("form") == "simple"))
    man = {"builder": "build_powered.py v1", "today": TODAY, "ages": AGES, "requests": REQUESTS,
           "sources": {"lapse_stimuli": str(ROOT / "data/stimuli_v2.jsonl"), "lapse_stimuli_sha256": sha(ROOT / "data/stimuli_v2.jsonl"),
                       "r2_generated_inputs": str(R2DIR / "generated/inputs.jsonl"), "r2_generated_inputs_sha256": sha(R2DIR / "generated/inputs.jsonl"),
                       "r1_generated_inputs_sha256": sha(R2DIR / "generated-r1/inputs.jsonl"), "r2_builder_sha256": sha(R2DIR / "build_inputs.py"),
                       "e2e_source": str(e2e_src), "e2e_source_sha256": sha(e2e_src)},
           "counts": {"lapse_contents": len(lapse), "r2_contents": len(r2), "new_contents": len(lapse) - len(r2),
                      "target_pairs_per_age": per_age(targets), "target_contents": len({r["content_key"] for r in targets}),
                      "excluded_r2_v0_items": len(excluded), "sweep_pairs_per_age": per_age(sweep),
                      "controls": dict(Counter(f"{r['control']}@{r['age_tag']}" for r in controls)),
                      "e2e_pairs_per_age": dict(Counter(f"{r['e2e_subset']}@{r['age_tag']}" for r in e2e if r["form"] == "simple")),
                      "e2e_anchor_redated_pairs_M8": sum(r["anchor_redated"] for r in e2e if r["form"] == "simple" and r["age_tag"] == "M8"),
                      "e2e_anchor_redated_pairs_M1": sum(r["anchor_redated"] for r in e2e if r["form"] == "simple" and r["age_tag"] == "M1")},
           "excluded_r2_v0_contents": excluded, "e2e_dropped_no_tool_frame": e2e_dropped,
           "hashes": {f"{n}.jsonl": sha(gen / f"{n}.jsonl") for n in files}}
    (gen / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(json.dumps(man["counts"], indent=1)); print(json.dumps(man["hashes"], indent=1)); print("e2e dropped:", len(e2e_dropped))


if __name__ == "__main__":
    main()

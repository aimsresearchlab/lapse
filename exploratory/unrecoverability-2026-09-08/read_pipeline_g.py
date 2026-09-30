"""Part G reader step (Graphiti + Letta) for research/UNRECOVERABILITY_SPEC_2026-09-08.md.

Builds reader items from writer_trace_graphiti_deepseek_official.jsonl and
writer_trace_letta_openrouter-deepseek-deepseek-v4-flash.jsonl, then runs
them through the Part A reader format (run_ladder.call/parse).

Arms: G-TEXT-{PROG,SIMPLE,BOUND}, G-FIELDS-{...}, L-TEXT-{PROG,SIMPLE,BOUND}.
    python read_pipeline_g.py --build | --dry | --full [--readers ...]
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
import concurrent.futures as cf
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "read-time-ladder-2026-09-08"))
from run_ladder import DIRECT, NOTES, READERS, SYS, Q_A1, Q_A2, call, parse  # noqa: E402
from read_pipeline import label_writer  # noqa: E402
from openai import OpenAI  # noqa: E402

TODAY, WRITTEN = "2026-09-08", "2026-01-06"
G_TRACE = HERE / "writer_trace_graphiti_deepseek_official.jsonl"
L_TRACE = HERE / "writer_trace_letta_openrouter-deepseek-deepseek-v4-flash.jsonl"


def g_witness_edge(r):
    """Edge whose fact carries the witness token and mentions the user (first such)."""
    tok = r["wit_token"].lower()
    cands = [e for e in r["edges"] if tok in (e["fact"] or "").lower()]
    user_first = [e for e in cands if "user" in (e["fact"] or "").lower()]
    return (user_first or cands or [None])[0]


def g_fields(e):
    va = (e["valid_at"] or "")[:10] or "not set"
    ia = (e["invalid_at"] or "")[:10] or "not set"
    return f"{e['fact']}\nvalid from: {va}; valid until: {ia}"


def l_witness_line(r):
    """Line of any memory block (or archival passage) that carries the witness token."""
    tok = r["wit_token"].lower()
    for b in r["blocks_after"]:
        for line in b["value"].splitlines():
            if tok in line.lower():
                return line.strip()
    for p in r["passages"]:
        if tok in p.lower():
            return p.strip()
    return None


def build():
    items, stats = [], {"g_dropped": [], "l_dropped": [], "g_missing": 0, "l_missing": 0}
    G = {json.loads(l)["id"]: json.loads(l) for l in G_TRACE.read_text().splitlines()} if G_TRACE.exists() else {}
    L = {json.loads(l)["id"]: json.loads(l) for l in L_TRACE.read_text().splitlines()} if L_TRACE.exists() else {}
    for frame in NOTES:
        label = NOTES[frame][2]
        for cl in range(12):
            for form in ("PROG", "SIMPLE", "BOUND"):
                uid = f"U-{frame}-{cl:02d}-{form}"
                notes = {}
                g = G.get(uid)
                if g is None: stats["g_missing"] += 1
                elif g["error"] or not g_witness_edge(g): stats["g_dropped"].append(uid)
                else:
                    e = g_witness_edge(g)
                    notes[f"G-TEXT-{form}"] = e["fact"]
                    notes[f"G-FIELDS-{form}"] = g_fields(e)
                l = L.get(uid)
                if l is None: stats["l_missing"] += 1
                elif l["error"] or not l_witness_line(l): stats["l_dropped"].append(uid)
                else:
                    notes[f"L-TEXT-{form}"] = l_witness_line(l)
                for arm, note in notes.items():
                    wl = label_writer(note.split("\n")[0])
                    for task in ("A1", "A2"):
                        items.append({"id": f"R-{frame}-{cl:02d}-{arm}-{task}", "frame": frame, "cluster": cl,
                                      "arm": arm, "form": form, "task": task, "witness": g["witness"] if g else l["witness"],
                                      "wit_token": g["wit_token"] if g else l["wit_token"], "writer_label": wl,
                                      "written": WRITTEN, "today": TODAY, "note": note,
                                      "system": SYS.format(today=TODAY, written=WRITTEN, note=note),
                                      "query": Q_A1 if task == "A1" else Q_A2.format(label=label)})
    return items, stats


def run(reader, items, limit, workers):
    tag = re.sub(r"[^a-z0-9]+", "-", reader.lower())
    if reader in DIRECT:
        base, keyenv, _, _ = DIRECT[reader]; client, tag = OpenAI(base_url=base, api_key=os.environ[keyenv]), tag + "-official"
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out_path = HERE / f"reader_g_trace_{tag}.jsonl"
    done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()} if out_path.exists() and limit is None else set()
    todo = [it for it in items if it["id"] not in done][: limit or None]
    out = open(out_path if limit is None else os.devnull, "a")
    def one(it):
        t0 = time.time()
        try: resp, meta = call(client, reader, it); err = None
        except Exception as e: resp, meta, err = "", {}, repr(e)  # noqa: BLE001
        rec = {k: it[k] for k in ("id", "frame", "cluster", "arm", "form", "task", "witness", "wit_token", "writer_label", "note")}
        rec.update(reader=reader, response=resp, error=err, **meta, secs=round(time.time() - t0, 1), **parse(it, resp)); return rec
    n = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            n += 1; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if limit or n % 100 == 0 or rec["error"]:
                print(f"[{tag}] {n}/{len(todo)} {rec['id']} -> {rec['response'][:60]!r} {rec.get('parse')}{' ERR ' + rec['error'] if rec['error'] else ''}", file=sys.stderr)
    out.close(); print(f"[{tag}] done {n} (skipped {len(done)})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--readers", nargs="*", default=READERS); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items, stats = build()
    (HERE / "reader_g_inputs.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    print(f"{len(items)} items/reader; graphiti dropped {len(stats['g_dropped'])} missing {stats['g_missing']}; letta dropped {len(stats['l_dropped'])} missing {stats['l_missing']}", file=sys.stderr)
    if a.build: return
    for r in a.readers: run(r, items, 4 if a.dry else None, a.workers)


if __name__ == "__main__":
    main()

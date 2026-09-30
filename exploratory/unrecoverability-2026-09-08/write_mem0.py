"""Writer step of research/UNRECOVERABILITY_SPEC_2026-09-08.md.

288 grid utterances (96 witnesses x PROG/SIMPLE/BOUND) through installed
mem0 2.0.19, one isolated user per item, DeepSeek official endpoint.
The clock is patched to 2026-01-06 BEFORE mem0 is imported so the
extraction prompt's "Today's date" and mem0's created_at match the stale
write date used by the reader.

    <venv>/bin/python write_mem0.py
"""
from __future__ import annotations

import datetime as _dt

WRITE_DATE = _dt.date(2026, 1, 6)


class _FakeDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        base = cls(WRITE_DATE.year, WRITE_DATE.month, WRITE_DATE.day, 12, 0, 0)
        return base.replace(tzinfo=tz) if tz else base


_dt.datetime = _FakeDatetime  # must precede `import mem0`

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from mem0 import Memory  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "exploratory" / "read-time-ladder-2026-09-08"))
from run_ladder import NOTES, witnesses, bound_phrase  # noqa: E402

# grid c0 first-person utterance per frame: (prog, simple)
UTT = {
    "lodging":        ("I'm staying at {w} on Fifth Street", "I live at {w} on Fifth Street"),
    "workplace":      ("I'm working at {w}", "I work at {w}"),
    "vehicle":        ("I'm driving a {w}", "I drive a {w}"),
    "class":          ("I'm taking a ceramics class at {w} on Tuesdays", "I take a ceramics class at {w} on Tuesdays"),
    "equipment":      ("I'm using a {w} on loan from the conservatory", "I use a {w} on loan from the conservatory"),
    "affiliate_role": ("I'm lecturing at {w} as an affiliate", "I lecture at {w} as an affiliate"),
    "project":        ("I'm working on {w}", "I work on {w}"),
    "household":      ("I'm staying with my cousin {w}", "I live with my cousin {w}"),
}
CARRIER = "hey! quick life update — {core}. also, any podcast recommendations for my commute?"

cfg = {
    "llm": {"provider": "openai", "config": {
        "model": "deepseek-v4-flash",
        "openai_base_url": "https://api.deepseek.com",
        "api_key": os.environ["DEEPSEEK_API_KEY"],
        "temperature": 0.0,
    }},
    "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5"}},
    "vector_store": {"provider": "qdrant", "config": {
        "path": str(HERE / "qdrant_deepseek_official"), "embedding_model_dims": 384, "on_disk": True}},
}


def build() -> list[dict]:
    items = []
    bound = bound_phrase(WRITE_DATE)
    for frame, (prog, simple) in UTT.items():
        for cl, (w, tok) in enumerate(witnesses()[frame]):
            cores = {"PROG": prog.format(w=w), "SIMPLE": simple.format(w=w),
                     "BOUND": prog.format(w=w) + " " + bound}
            for form, core in cores.items():
                items.append({"id": f"U-{frame}-{cl:02d}-{form}", "frame": frame, "cluster": cl,
                              "form": form, "witness": w, "wit_token": tok,
                              "text": CARRIER.format(core=core)})
    return items


def main() -> None:
    items = build()
    (HERE / "writer_inputs.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    out_path = HERE / "writer_trace_deepseek_official.jsonl"
    done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    m = Memory.from_config(cfg)
    out = out_path.open("a")
    for i, it in enumerate(items):
        if it["id"] in done:
            continue
        uid = f"u-{it['id']}"
        t0 = time.time()
        err = None
        try:
            r = m.add([{"role": "user", "content": it["text"]}], user_id=uid)
            mems = [x["memory"] for x in m.get_all(filters={"user_id": uid})["results"]]
        except Exception as e:  # noqa: BLE001
            r, mems, err = None, [], repr(e)
        wit = [x for x in mems if it["wit_token"].lower() in x.lower()]
        out.write(json.dumps(dict(it, writer="deepseek-v4-flash@api.deepseek.com", write_date=WRITE_DATE.isoformat(),
                                  add=r, memories=mems, witness_memory=wit[0] if wit else None,
                                  n_witness_memories=len(wit), error=err, secs=round(time.time() - t0, 1)),
                             ensure_ascii=False) + "\n")
        out.flush()
        print(f"{i + 1}/{len(items)} {it['id']} -> {wit[0] if wit else 'DROPPED'}" + (f" ERR {err}" if err else ""), file=sys.stderr)
    out.close()


if __name__ == "__main__":
    main()

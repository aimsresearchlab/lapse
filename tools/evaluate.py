"""Report the paper's per-model LAPSE statistics for any scored trace.

Runs the same per-column tests as tools/analysis_v2.py (which is frozen and
hard-codes the paper's columns): the progressive-vs-simple McNemar on matched
memory notes, the carrier sign test, the same-lexeme check, per-frame
flattening, and non-commitment by gap. The cross-model confirmatory family
is omitted because it is defined only for the paper's three models.

Usage: python3 tools/evaluate.py traces/<run>.scored.jsonl
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_v2  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".scored.jsonl"):
        sys.exit(__doc__.strip().splitlines()[-1])
    path = Path(sys.argv[1]).resolve()
    if path.parent != analysis_v2.ROOT / "traces":
        sys.exit(f"scored trace must be in {analysis_v2.ROOT / 'traces'}")
    analysis_v2.COLUMNS = [("model", path.name.removesuffix(".scored.jsonl"))]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analysis_v2.main()
    print(buf.getvalue().split("\n" + "=" * 72 + "\nCONFIRMATORY FAMILY")[0])


if __name__ == "__main__":
    main()

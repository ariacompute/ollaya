"""Packing goldens for afm-de-latest vs afm_d.de.model.build_sequence (when afm_d is importable)."""
from __future__ import annotations

import argparse
import json
import os


CASES = [
    {
        "id": "noul_short",
        "state": "refund please",
        "q": {"t": "noul", "ins": "Is a refund requested?", "opts": [
            "false: no, the statement does not hold",
            "true: yes, the statement holds",
        ]},
    },
    {
        "id": "long_state_tail",
        "state": ("prefix " * 400) + "DECISIVE: cancel now",
        "q": {"t": "choice", "ins": "Intent?", "opts": ["cancel: leave", "keep: stay"]},
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/afm-de/goldens-packing.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    try:
        from afm_d.de.model import build_sequence
        from transformers import AutoTokenizer
        from . import ref
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        with open(a.out, "w") as f:
            for c in CASES:
                q = {"t": c["q"]["t"], "ins": c["q"]["ins"], "opts": c["q"]["opts"]}
                # build_sequence expects opts via render_options path; use internal form
                ids, markers = build_sequence(
                    tok, c["state"],
                    {"t": q["t"], "ins": q["ins"], "opts": q["opts"]},
                    max_len=ref.MAX_LEN, head_max_len=ref.HEAD_MAX_LEN, truncate_left=True,
                )
                f.write(json.dumps({"id": c["id"], "ids": ids, "markers": markers}) + "\n")
        print("wrote", a.out)
    except ImportError as e:
        print("skip goldens (afm_d not importable):", e)


if __name__ == "__main__":
    main()

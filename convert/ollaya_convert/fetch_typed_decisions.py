"""Download LocalLLaMA/typed-decisions test split into out/data/typed-decisions-test.parquet.

Goldens / export_llama / parity read this file via ollaya_convert.cases.TYPED_DECISIONS.

    cd convert
    uv run python -m ollaya_convert.fetch_typed_decisions
"""
from __future__ import annotations

import os
import sys
import urllib.request

# Same path as ollaya_convert.cases.TYPED_DECISIONS
DEST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "out", "data", "typed-decisions-test.parquet")
)
# Official HF parquet for the combined test split (400 states).
URL = (
    "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/"
    "all/test-00000-of-00001.parquet"
)


def main() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.isfile(DEST) and os.path.getsize(DEST) > 0 and "--force" not in sys.argv:
        print("already present:", DEST, f"({os.path.getsize(DEST)} bytes)")
        print("pass --force to re-download")
        return
    print("downloading", URL)
    tmp = DEST + ".part"
    urllib.request.urlretrieve(URL, tmp)
    os.replace(tmp, DEST)
    n_rows = "?"
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(DEST)
        need = {"id", "state", "questions"}
        cols = set(table.column_names)
        missing = need - cols
        if missing:
            os.unlink(DEST)
            sys.exit(f"error: parquet missing columns {sorted(missing)}; got {sorted(cols)}")
        n_rows = table.num_rows
    except ImportError:
        pass
    print(f"wrote {DEST} ({n_rows} rows, {os.path.getsize(DEST)} bytes)")


if __name__ == "__main__":
    main()

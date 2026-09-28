#!/usr/bin/env bash
# Export and package AFM-D models into ollaya/registry (requires local checkpoints / GGUF).
#
# Encoder:
#   AFM_DE_CHECKPOINT=~/models/afm-de ./scripts/package-afm.sh de
# Decoder (after model/afm-d/scripts/merge_dd_gguf.sh + Hub upload):
#   AFM_DD_COMMIT=<sha> ./scripts/package-afm.sh dd
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/convert"
TRACK="${1:-both}"

if [[ "$TRACK" == "de" || "$TRACK" == "both" ]]; then
  CKPT="${AFM_DE_CHECKPOINT:-$HOME/models/afm-de}"
  test -f "$CKPT/model.safetensors" || { echo "missing $CKPT/model.safetensors"; exit 1; }
  uv run python -m ollaya_convert.families.afm_de.export --checkpoint "$CKPT" --out out/afm-de
  # optional: derive fp16 into out/afm-de-fp16 (reuse existing fp16.py patterns)
  AFM_DE_CHECKPOINT="$CKPT/model.safetensors" uv run python -m ollaya_convert.package afm-de
fi

if [[ "$TRACK" == "dd" || "$TRACK" == "both" ]]; then
  test -n "${AFM_DD_COMMIT:-}" || { echo "set AFM_DD_COMMIT to the Hub commit of the Q8_0 GGUF"; exit 1; }
  echo "Run export_llama afm-dd against the local GGUF, then:"
  echo "  AFM_DD_COMMIT=$AFM_DD_COMMIT uv run python -m ollaya_convert.package afm-dd"
fi

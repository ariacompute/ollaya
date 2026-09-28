#!/usr/bin/env bash
# Export and package AFM-D models into ollaya/registry (requires local checkpoints / GGUF).
#
# Encoder:
#   AFM_DE_CHECKPOINT=~/.ariacompute/models/afm-de ./scripts/package-afm.sh de
# Decoder (after model/afm-d/scripts/merge_dd_gguf.sh + Hub upload of a *public* GGUF):
#   AFM_DD_COMMIT=<sha> ./scripts/package-afm.sh dd
#
# Derived blob URLs default to this fork's GitHub-hosted registry tree (see DEFAULT_REGISTRY).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/convert"
TRACK="${1:-both}"
ORIGIN="${SITE_ORIGIN:-https://raw.githubusercontent.com/ariacompute/ollaya/main/registry}"

if [[ "$TRACK" == "de" || "$TRACK" == "both" ]]; then
  CKPT="${AFM_DE_CHECKPOINT:-$HOME/.ariacompute/models/afm-de}"
  test -f "$CKPT/model.safetensors" || { echo "missing $CKPT/model.safetensors"; exit 1; }
  uv run python -m ollaya_convert.families.afm_de.export --checkpoint "$CKPT" --out out/afm-de
  # optional: derive fp16 into out/afm-de-fp16 (reuse existing fp16.py patterns)
  AFM_DE_CHECKPOINT="$CKPT/model.safetensors" \
    uv run python -m ollaya_convert.package afm-de --origin "$ORIGIN"
fi

if [[ "$TRACK" == "dd" || "$TRACK" == "both" ]]; then
  test -n "${AFM_DD_COMMIT:-}" || { echo "set AFM_DD_COMMIT to the Hub commit of the Q8_0 GGUF"; exit 1; }
  echo "Run export_llama afm-dd against the local GGUF, then:"
  echo "  AFM_DD_COMMIT=$AFM_DD_COMMIT uv run python -m ollaya_convert.package afm-dd --origin $ORIGIN"
fi

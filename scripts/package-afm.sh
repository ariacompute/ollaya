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
  GGUF="${AFM_DD_GGUF:-/path/to/afm-dd-2b-Q8_0.gguf}"
  SERVER="${AFM_DD_LLAMA_SERVER:-/path/to/llama-server}"
  cat <<EOF
Decoder (afm-dd): run export_llama against the local GGUF, then package.

  uv run python -m ollaya_convert.families.llm_common.export_llama afm-dd \\
    --server ${SERVER} --gguf ${GGUF} --slug 2b-q8_0 \\
    --repo ariacompute/afm-dd --revision ${AFM_DD_COMMIT} --file gguf/afm-dd-2b-Q8_0.gguf \\
    --n-ctx 4096 --temperature 1.0

  AFM_DD_COMMIT=${AFM_DD_COMMIT} uv run python -m ollaya_convert.package afm-dd --origin ${ORIGIN}

Optional env: AFM_DD_GGUF, AFM_DD_LLAMA_SERVER (defaults shown as placeholders above).
EOF
fi

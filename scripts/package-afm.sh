#!/usr/bin/env bash
# Export and package AFM-D models into ollaya/registry.
#
# Encoder: the pinned checkpoint is downloaded on first run into AFM_DE_CHECKPOINT
# (default ~/.ariacompute/models/afm-de), so a fresh checkout needs no manual copy:
#   ./scripts/package-afm.sh de
#   OLLAYA_HUB=modelscope ./scripts/package-afm.sh de
#   AFM_DE_CHECKPOINT=~/.ariacompute/models/afm-de ./scripts/package-afm.sh de
# Decoder (after model/afm-d/scripts/merge_dd_gguf.sh + Hub upload of a *public* GGUF):
#   AFM_DD_COMMIT=<sha> ./scripts/package-afm.sh dd
#
# The convert/ project pins torch to the CUDA 13 wheel index (see pyproject.toml), which pulls
# GBs of nvidia-* wheels on Linux. Export runs on the CPU, so by default this script uses a
# small cached venv with the CPU torch wheel and only the packages the export/package path
# imports; AFM_DE_FULL_SYNC=1 falls back to `uv run` on the full project environment.
#
# Derived blob URLs default to this fork's GitHub-hosted registry tree (see DEFAULT_REGISTRY).
#
# Environment:
#   AFM_DE_CHECKPOINT     checkpoint directory (default ~/.ariacompute/models/afm-de)
#   OLLAYA_HUB            huggingface | modelscope | auto (default: HF, then ModelScope)
#   AFM_DE_URL_BASE       override the primary Hub base URL (mirror / offline copy)
#   AFM_DE_SHA256         override the expected sha256 of model.safetensors
#   AFM_DE_NO_DOWNLOAD=1  never download; require a complete local checkpoint
#   AFM_DE_FULL_SYNC=1    use `uv run` on the full convert/ environment (CUDA wheels)
#   AFM_DE_VENV           light venv path (default ~/.cache/ollaya/afm-de-cpu)
#   AFM_DE_PYTHON         Python for the light venv (default: python3.12, else uv-managed 3.12)
#   SITE_ORIGIN           origin for the derived blob URLs
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/package-afm.sh [de|dd|both]   (default: both)
       ./scripts/package-afm.sh --help

Commands:
  de    Download (once) the pinned ariacompute/afm-de checkpoint, export the
        fp32 ONNX graph, derive its fp16 variant, and package the derived
        registry blobs and manifests for afm-de.
  dd    With AFM_DD_COMMIT set, print the decoder export + packaging commands
        for the merged Q8_0 GGUF (the script does not run them).
  both  Run de, then print the dd instructions.

Encoder checkpoint (default ~/.ariacompute/models/afm-de):
  AFM_DE_CHECKPOINT     checkpoint directory
  OLLAYA_HUB            huggingface | modelscope | auto (HF first, MS fallback)
  AFM_DE_URL_BASE       override the primary Hub base URL (mirror / offline copy)
  AFM_DE_SHA256         override the expected sha256 of model.safetensors
  AFM_DE_NO_DOWNLOAD=1  never download; require a complete local checkpoint

Python environment (export runs on the CPU):
  By default a small cached venv at ~/.cache/ollaya/afm-de-cpu holds only
  torch==2.14.0+cpu, onnx, onnxscript, numpy, transformers, tokenizers, laya
  and CPU-only onnxruntime (for the fp16 graph); no CUDA / pyarrow / accelerate.
  AFM_DE_FULL_SYNC=1    use `uv run` on the full convert/ project (CUDA wheels)
  AFM_DE_VENV           override the light venv path
  AFM_DE_PYTHON         Python for the venv (default python3.12, else uv-managed 3.12)

Registry / decoder:
  SITE_ORIGIN           origin for derived blob URLs (default: GitHub raw registry)
  AFM_DD_COMMIT         Hub commit of the public Q8_0 GGUF (required for dd)
  AFM_DD_GGUF           path to afm-dd-2b-Q8_0.gguf
  AFM_DD_LLAMA_SERVER   path to the llama-server binary

Examples:
  ./scripts/package-afm.sh de
  OLLAYA_HUB=modelscope ./scripts/package-afm.sh de
  AFM_DE_FULL_SYNC=1 ./scripts/package-afm.sh de
  AFM_DD_COMMIT=<sha> ./scripts/package-afm.sh dd

The dd path needs the full uv environment (pyarrow, llama-server); the light
venv is for the encoder only.
EOF
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/convert"
TRACK="${1:-both}"
ORIGIN="${SITE_ORIGIN:-https://raw.githubusercontent.com/ariacompute/ollaya/main/registry}"

# Keep in sync with catalog.py's AFM_DE_COMMIT and the published manifest's weights digest:
# package.py refuses a checkpoint whose sha256 differs from the pinned upstream file.
AFM_DE_COMMIT="5c570f648946de7879b7f2a2634f07e2e5b06180"
AFM_DE_SHA256="${AFM_DE_SHA256:-24c547bb5497396ff235070e7370e0dd254362393efbbab0f5d29f3e60afda06}"
HF_BASE="${AFM_DE_URL_BASE:-https://huggingface.co/ariacompute/afm-de/resolve/${AFM_DE_COMMIT}}"
MS_BASE="https://www.modelscope.cn/models/AriaCompute/afm-de/resolve/master"
# Everything afm_de.export reads: laya.load() needs the weights + encoder config, and the
# exporter copies tokenizer/tokenizer.json and reads rl_agent_config.json.
AFM_DE_FILES=(model.safetensors rl_agent_config.json encoder/config.json
              tokenizer/tokenizer.json tokenizer/tokenizer_config.json)

case "${OLLAYA_HUB:-auto}" in
  huggingface) HUB_BASES=("$HF_BASE") ;;
  modelscope)  HUB_BASES=("$MS_BASE") ;;
  *)           HUB_BASES=("$HF_BASE" "$MS_BASE") ;;
esac

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo ""
  fi
}

# Fetch one file into the checkpoint, atomically (.part -> final), trying each hub in turn.
fetch_checkpoint_file() {
  local rel="$1" dir="$2" dest tmp base first=1
  dest="$dir/$rel"
  tmp="$dest.part"
  [[ -s "$dest" ]] && return 0
  mkdir -p "$(dirname "$dest")"
  for base in "${HUB_BASES[@]}"; do
    if [[ "$first" != 1 ]]; then rm -f "$tmp"; fi  # never mix bytes from two hubs
    first=0
    echo "downloading $rel from $base" >&2
    if command -v curl >/dev/null 2>&1; then
      if [[ -s "$tmp" ]]; then
        curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -C - -o "$tmp" "$base/$rel" || continue
      else
        curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$tmp" "$base/$rel" || continue
      fi
    elif command -v wget >/dev/null 2>&1; then
      if [[ -s "$tmp" ]]; then
        wget -q -c -O "$tmp" "$base/$rel" || continue
      else
        wget -q -O "$tmp" "$base/$rel" || continue
      fi
    else
      echo "need curl or wget to download $rel" >&2
      return 1
    fi
    [[ -s "$tmp" ]] || continue
    mv "$tmp" "$dest"
    echo "  -> $dest" >&2
    return 0
  done
  rm -f "$tmp"
  echo "could not download $rel from: ${HUB_BASES[*]}" >&2
  return 1
}

verify_afm_de_weights() {
  local sum
  sum="$(sha256_of "$1")"
  if [[ -z "$sum" ]]; then
    echo "warning: no sha256sum/shasum found; skipping weights checksum" >&2
    return 0
  fi
  if [[ "$sum" != "$AFM_DE_SHA256" ]]; then
    echo "$1: sha256 $sum, expected $AFM_DE_SHA256" >&2
    echo "the registry pins ariacompute/afm-de@$AFM_DE_COMMIT; remove the file to re-download," >&2
    echo "or set AFM_DE_SHA256 / AFM_DE_URL_BASE for another upstream revision." >&2
    return 1
  fi
}

# Only what afm_de.export / fp16.py / ollaya_convert.package import: torch (CPU), onnx,
# onnxscript, numpy, laya (transformers / tokenizers / safetensors) and the CPU-only
# onnxruntime wheel for fp16.py's float16 converter. Still no pyarrow, accelerate or
# nvidia-* wheels; versions mirror convert/pyproject.toml + uv.lock.
AFM_DE_VENV="${AFM_DE_VENV:-${XDG_CACHE_HOME:-$HOME/.cache}/ollaya/afm-de-cpu}"
if [[ -n "${AFM_DE_PYTHON:-}" ]]; then
  PY_SPEC="$AFM_DE_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  PY_SPEC="python3.12"
else
  PY_SPEC="3.12"  # uv downloads a managed Python 3.12 when the system has none
fi
AFM_DE_LAYA_URL="https://files.pythonhosted.org/packages/b5/fd/dd60f49451ef43af50c3fcd5d5d9b12aea590f5bcb9afb3e5ec5171b6fa6/laya-0.3.7.tar.gz"
AFM_DE_DEPS=(
  "torch==2.14.0+cpu"
  "onnx==1.23.*"
  "onnxruntime==1.30.*"
  "onnxscript==0.7.*"
  "numpy>=2"
  "transformers==5.17.*"
  "tokenizers==0.23.*"
  "laya @ ${AFM_DE_LAYA_URL}"
)

ensure_light_env() {
  command -v uv >/dev/null 2>&1 || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
  local help spec stamp
  help="$(uv pip install --help 2>/dev/null)"
  if [[ "$help" != *"--torch-backend"* ]]; then
    echo "uv is too old for --torch-backend; run: uv self update" >&2
    exit 1
  fi
  if [[ ! -x "$AFM_DE_VENV/bin/python" ]]; then
    mkdir -p "$(dirname "$AFM_DE_VENV")"
    echo "creating light venv $AFM_DE_VENV (Python $PY_SPEC)" >&2
    uv venv --python "$PY_SPEC" "$AFM_DE_VENV" >&2 || exit 1
  fi
  stamp="$AFM_DE_VENV/.afm-de-deps"
  spec="$(printf '%s\n' "${AFM_DE_DEPS[@]}")"
  if [[ "$(cat "$stamp" 2>/dev/null || true)" == "$spec" ]]; then
    return 0  # already installed: no network needed on repeat runs
  fi
  echo "installing CPU-only export dependencies into $AFM_DE_VENV" >&2
  uv pip install --python "$AFM_DE_VENV/bin/python" --torch-backend cpu "${AFM_DE_DEPS[@]}" >&2 || exit 1
  printf '%s' "$spec" > "$stamp"
}

if [[ "${AFM_DE_FULL_SYNC:-}" == "1" ]]; then
  PYCMD=("uv" "run" "python")
else
  PYCMD=("$AFM_DE_VENV/bin/python")
fi

if [[ "$TRACK" == "de" || "$TRACK" == "both" ]]; then
  CKPT="${AFM_DE_CHECKPOINT:-$HOME/.ariacompute/models/afm-de}"
  mkdir -p "$CKPT"
  for rel in "${AFM_DE_FILES[@]}"; do
    if [[ -s "$CKPT/$rel" ]]; then continue; fi
    if [[ "${AFM_DE_NO_DOWNLOAD:-}" == "1" ]]; then
      echo "missing $CKPT/$rel (AFM_DE_NO_DOWNLOAD=1)" >&2
      exit 1
    fi
    fetch_checkpoint_file "$rel" "$CKPT" || exit 1
  done
  verify_afm_de_weights "$CKPT/model.safetensors" || exit 1
  if [[ "${AFM_DE_FULL_SYNC:-}" != "1" ]]; then ensure_light_env; fi
  "${PYCMD[@]}" -m ollaya_convert.families.afm_de.export --checkpoint "$CKPT" --out out/afm-de
  # catalog.py ships fp32 + fp16 graph layers for afm-de: derive the fp16 variant first.
  "${PYCMD[@]}" -m ollaya_convert.fp16 out/afm-de out/afm-de-fp16
  AFM_DE_CHECKPOINT="$CKPT/model.safetensors" \
    "${PYCMD[@]}" -m ollaya_convert.package afm-de --origin "$ORIGIN"
fi

if [[ "$TRACK" == "dd" || "$TRACK" == "both" ]]; then
  test -n "${AFM_DD_COMMIT:-}" || { echo "set AFM_DD_COMMIT to the Hub commit of the Q8_0 GGUF"; exit 1; }
  GGUF="${AFM_DD_GGUF:-/path/to/afm-dd-2b-Q8_0.gguf}"
  SERVER="${AFM_DD_LLAMA_SERVER:-/path/to/llama-server}"
  cat <<EOF
Decoder (afm-dd): run these from convert/ (uv project root; pyproject.toml lives there).
The light CPU venv above is for the encoder only; the decoder path needs the full convert/
environment (pyarrow for the parquet, llama-server export), so these commands use uv run.

  cd ${ROOT}/convert
  uv run python -m ollaya_convert.fetch_typed_decisions   # → out/data/typed-decisions-test.parquet
  uv run python -m ollaya_convert.families.llm_common.export_llama afm-dd \\
    --server ${SERVER} --gguf ${GGUF} --slug 2b-q8_0 \\
    --repo ariacompute/afm-dd --revision ${AFM_DD_COMMIT} --file gguf/afm-dd-2b-Q8_0.gguf \\
    --n-ctx 4096 --temperature 1.0 \\
    --device cpu   # or CUDA0 when GPU + CUDA llama-server work

  AFM_DD_COMMIT=${AFM_DD_COMMIT} uv run python -m ollaya_convert.package afm-dd --origin ${ORIGIN}

Optional env: AFM_DD_GGUF, AFM_DD_LLAMA_SERVER (defaults shown as placeholders above).
--server must be the llama-server binary, not the llama.cpp source tree.
EOF
fi

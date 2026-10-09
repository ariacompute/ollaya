# afm-dd (`afm-dd-latest`)

**AFM-D Decoder** is AriaCompute's causal typed-decision track: SemIf `direct` option-letter
logits on MiniCPM5-2B (± PEFT LoRA), shipped as a merged Q8_0 GGUF in the same Hub repo as the
adapter.

| | |
|---|---|
| Hub | [ariacompute/afm-dd](https://huggingface.co/ariacompute/afm-dd) · [AriaCompute/afm-dd](https://www.modelscope.cn/models/AriaCompute/afm-dd) |
| Layout | `afm-dd-latest` |
| Engine | llama.cpp (`cold` plan) |
| Prompt | SemIf SYSTEM + JSON user; **MiniCPM BOS** |
| Options | A–P (2–16) |

Dual-hub weight URLs; see [afm-de](afm-de.md) for `OLLAYA_HUB`.

## Upstream GGUF

```bash
cd ../model/afm-d && ./scripts/merge_dd_gguf.sh
# upload out/afm-dd/gguf/afm-dd-2b-Q8_0.gguf as gguf/afm-dd-2b-Q8_0.gguf on HF + ModelScope
```

## Convert

`uv run` must be from `convert/` (the uv project root). `--server` is the `llama-server` binary, not the llama.cpp tree.

```bash
cd convert
uv run python -m ollaya_convert.families.llm_common.export_llama afm-dd \
  --server /path/to/llama-server --gguf .../afm-dd-2b-Q8_0.gguf --slug 2b-q8_0 \
  --repo ariacompute/afm-dd --revision <sha> --file gguf/afm-dd-2b-Q8_0.gguf \
  --n-ctx 4096 --temperature 1.0
uv run python -m ollaya_convert.package afm-dd
```

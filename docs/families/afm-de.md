# afm-de (`afm-de-latest`)

**AFM-D Encoder** is AriaCompute's non-autoregressive typed-decision track: a Laya-layout
ModernBERT-large DecisionModel, RLCD-fine-tuned from `convaiinnovations/laya`.

| | |
|---|---|
| Hub | [ariacompute/afm-de](https://huggingface.co/ariacompute/afm-de) · [AriaCompute/afm-de](https://www.modelscope.cn/models/AriaCompute/afm-de) |
| Layout | `afm-de-latest` |
| Engine | ONNX (`OnnxModel`, same DecisionModel I/O as Laya) |
| Context | `max_len` 1024, `head_max_len` 512 |
| Options | `option_desc_max` 96; long states keep the **tail** |
| High-K Choice | shortlist (`threshold` 40, `k` 20) |
| Decision T | fixed 1.0 |

Weights are foreign layers with **dual URLs** (Hugging Face + ModelScope). `ollaya pull` picks the
hub via `OLLAYA_HUB` (`huggingface` | `modelscope` | `auto`; `auto` prefers ModelScope for `zh*` locales).

## Convert

```bash
# snapshot Hub → local, then:
uv run python -m ollaya_convert.families.afm_de.export \
  --checkpoint /path/to/afm-de --out out/afm-de
# optional fp16 derivative, then:
uv run python -m ollaya_convert.package afm-de
```

## Parity

Packing goldens vs `afm_d.de.model.build_sequence`. ORT fp32 vs PyTorch AFM-D at T=1.0.

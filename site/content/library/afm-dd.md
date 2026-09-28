AFM-D Decoder is AriaCompute's SemIf-style causal decision model on MiniCPM5-2B: it reads
option-letter logits (A–P) after a typed JSON prompt. Up to 16 options per question.

Pull with `ollaya pull afm-dd` (alias `afm-dd:2b`). The GGUF lives in the same Hub repository as
the LoRA adapter (`ariacompute/afm-dd` / `AriaCompute/afm-dd`). Use `OLLAYA_HUB` to choose the
download mirror.

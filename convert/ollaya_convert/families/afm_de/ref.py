"""Reference packing for `afm-de-latest` (AFM-D Encoder).

Mirrors `afm_d.de.model.build_sequence`: OPTION_DESC_MAX=96, truncate_left (tail),
max_len=1024, head_max_len=512. Hub: ariacompute/afm-de / AriaCompute/afm-de.
"""
from __future__ import annotations

LAYOUT = "afm-de-latest"
FAMILY = "afm-de"
OPTION_DESC_MAX = 96
MAX_LEN = 1024
HEAD_MAX_LEN = 512
STATE_TRUNCATION = "tail"
SHORTLIST = {"threshold": 40, "k": 20}
HF_REPO = "ariacompute/afm-de"
HF_COMMIT = "402b7184eed0c64fe94f8cc2dfb716936fe4e43d"
MS_REPO = "AriaCompute/afm-de"


def decision_json(special_tokens, encoder="answerdotai/ModernBERT-large"):
    return {
        "engine": "onnx",
        "family": FAMILY,
        "layout": LAYOUT,
        "encoder": encoder,
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "option_desc_max": OPTION_DESC_MAX,
        "state_truncation": STATE_TRUNCATION,
        "shortlist": SHORTLIST,
        "special_tokens": special_tokens,
        "inputs": ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        "outputs": ["logits", "act_logits"],
        "min_markers": 2,
        "hubs": {"huggingface": HF_REPO, "modelscope": MS_REPO},
    }

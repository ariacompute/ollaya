"""Export AFM-D Encoder to Ollaya ONNX + decision.json (`afm-de-latest`).

    uv run python -m ollaya_convert.families.afm_de.export \
        --checkpoint /path/to/afm-de/checkpoint --out out/afm-de

The checkpoint is Laya-layout (model.safetensors, rl_agent_config.json, tokenizer/).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import onnx
import torch

from . import ref


def export(checkpoint: str, out_dir: str) -> str:
    import laya
    from ollaya_convert.export import (
        INPUT_NAMES, MIN_MARKERS, OPSET, OUTPUT_NAMES, Graph,
        expand_attention_masks, expand_rotary_caches, prescale_attention,
    )

    os.makedirs(out_dir, exist_ok=True)
    agent = laya.load(checkpoint, subfolder=None, device="cpu")
    graph = Graph(agent.model)
    graph.eval()
    q, s, k = 2, 32, max(MIN_MARKERS, 2)
    args = (
        torch.zeros(q, s, dtype=torch.long),
        torch.ones(q, s, dtype=torch.long),
        torch.zeros(q, k, dtype=torch.long),
        torch.ones(q, k, dtype=torch.bool),
        torch.zeros(q, dtype=torch.long),
    )
    onnx_path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(
            graph, args, dynamo=True, opset_version=OPSET,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamic_shapes={
                "input_ids": {0: "q", 1: "s"},
                "attention_mask": {0: "q", 1: "s"},
                "marker_pos": {0: "q", 1: "k"},
                "marker_mask": {0: "q", 1: "k"},
                "qtype": {0: "q"},
            },
            optimize=False,
        )
    program.save(onnx_path, external_data=False)
    if OPSET >= 23:
        model = onnx.load(onnx_path, load_external_data=True)
        expand_attention_masks(model)
        expand_rotary_caches(model)
        prescale_attention(model)
        if os.path.exists(onnx_path + ".data"):
            os.remove(onnx_path + ".data")
        onnx.save(model, onnx_path, save_as_external_data=True, location="model.onnx.data")
    onnx.checker.check_model(onnx_path, full_check=False)

    tok = agent.tok
    src_tok = os.path.join(checkpoint, "tokenizer", "tokenizer.json")
    if not os.path.isfile(src_tok):
        src_tok = os.path.join(checkpoint, "tokenizer.json")
    shutil.copy(src_tok, os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(checkpoint, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    special = {
        "cls": tok.cls_token_id, "sep": tok.sep_token_id,
        "mask": tok.mask_token_id, "pad": tok.pad_token_id, "mask_text": tok.mask_token,
    }
    decision = ref.decision_json(special, encoder=cfg.get("encoder", "answerdotai/ModernBERT-large"))
    decision["opset"] = OPSET
    decision["max_len"] = int(cfg.get("max_len", ref.MAX_LEN))
    decision["head_max_len"] = int(cfg.get("head_max_len", ref.HEAD_MAX_LEN))
    calibration = {
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": cfg.get("temperature_by_options", {}),
        "source": "AFM-D Encoder: decision temperature fixed at 1.0",
    }
    with open(os.path.join(out_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    return onnx_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = export(a.checkpoint, a.out)
    print("wrote %s (%.0f MB)" % (path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()

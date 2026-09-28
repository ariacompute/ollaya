"""Reference for `afm-dd-latest`: SemIf direct + MiniCPM5 chat template (with BOS).

Same SYSTEM / JSON user payload as jevk5-v1 / SemIf; tokenization prepends BOS.
Hub: ariacompute/afm-dd / AriaCompute/afm-dd (PEFT + merged Q8_0 GGUF in-repo).
"""
from __future__ import annotations

# Reuse jevk5 compile helpers (byte-identical SemIf prompt text).
from ollaya_convert.families.jevk5.ref import (  # noqa: F401
    LETTERS,
    SYSTEM,
    TooManyOptions,
    compile_question,
    compile_request,
    render_state,
    user_message,
)

LAYOUT = "afm-dd-latest"
FAMILY = "afm-dd"
PRE = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ADD_BOS = True
HF_REPO = "ariacompute/afm-dd"
MS_REPO = "AriaCompute/afm-dd"
UPSTREAM = {
    "product": "AFM-D Decoder",
    "base": "openbmb/MiniCPM5-2B",
    "base_revision": "12a3808a956f869c767195e9266b59c4d21d92e2",
    "prompt": "semif_phase1.direct",
}


class AfmDdError(ValueError):
    pass


def prompt_text(state, criterion, texts) -> str:
    return PRE + user_message(state, criterion, texts) + POST


def token_ids(srv, user: str, *, add_bos: bool = True):
    """PRE/POST with special parsing; user without; optional leading BOS."""
    ids = []
    if add_bos:
        # Empty tokenize with add_special=True yields [bos] on MiniCPM/Llama.
        bos = srv.tokenize("", add_special=True, parse_special=False)
        if len(bos) == 1:
            ids.extend(bos)
        else:
            # Fallback: tokenize a single letter with BOS and drop the letter.
            x = srv.tokenize("x", add_special=True, parse_special=False)
            if len(x) == 2:
                ids.append(x[0])
    ids.extend(srv.tokenize(PRE, add_special=False, parse_special=True))
    ids.extend(srv.tokenize(user, add_special=False, parse_special=False))
    ids.extend(srv.tokenize(POST, add_special=False, parse_special=True))
    return ids

"""
Wraps a HuggingFace causal LM to serve exactly what an arithmetic coder needs:
a quantized probability distribution over the next token, one step at a time,
reproducible bit-for-bit between an encode pass and a decode pass.

Design note: encode() always knows the real next symbol; decode() doesn't.
Both must derive the *same* pdf from *only* the tokens already committed.
So the API is split into two calls per step:
    pdf = predictor.next_token_pdf(base, precision)   # look, don't advance
    predictor.advance(token_id)                        # commit a token, advance state
The caller (encoder or decoder) decides where `token_id` comes from.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TokenPredictor:
    """Incremental, KV-cached next-token distribution source."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        # CPU is the safe default: matmul on GPU is not always bit-reproducible
        # across runs, and reproducibility is the whole game for lossless coding.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()  # disables dropout etc. -- required for determinism

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if bos is None and eos is None:
            raise ValueError(
                f"{checkpoint}'s tokenizer has neither bos_token_id nor "
                "eos_token_id; need some fixed token to prime the first step."
            )
        self._start_token_id = bos if bos is not None else eos

        self._past = None
        self._next_logits = None

    def model_size_bytes(self) -> int:
        """Bytes of loaded parameters + buffers, at the dtype actually in use.

        This is the honest "size of the model used for compression" figure --
        what you'd have to already possess (or ship) to reproduce this result,
        the way the paper's own 'adjusted compression rate' accounts for it.
        """
        return self.model.get_memory_footprint()

    @torch.no_grad()
    def reset(self) -> None:
        """Prime the model on the start token. Call once before each message."""
        self._past = None
        input_ids = torch.tensor([[self._start_token_id]], device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        self._past = out.past_key_values
        self._next_logits = out.logits[0, -1]

    def next_token_pdf(self, base: int, precision: int) -> np.ndarray:
        """Quantized probability distribution over the next token.

        Does NOT advance state -- call this before you know (encode) or have
        decoded (decode) the actual next token, then call advance() after.
        """
        if self._next_logits is None:
            raise RuntimeError("Call reset() before requesting a pdf.")
        probs = torch.softmax(self._next_logits.double(), dim=-1).cpu().numpy()
        return _quantize_pdf(probs, base, precision)

    @torch.no_grad()
    def advance(self, token_id: int) -> None:
        """Commit `token_id` as the next symbol and update cached state."""
        if self._past is None:
            raise RuntimeError("Call reset() before advance().")
        input_ids = torch.tensor([[token_id]], device=self.device)
        out = self.model(
            input_ids=input_ids, past_key_values=self._past, use_cache=True
        )
        self._past = out.past_key_values
        self._next_logits = out.logits[0, -1]

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def max_context_length(self) -> int:
        """Caller should chunk input longer than this (paper does the same)."""
        return getattr(self.model.config, "max_position_embeddings", None) or getattr(
            self.model.config, "max_length", 2048
        )


def _quantize_pdf(probs: np.ndarray, base: int, precision: int) -> np.ndarray:
    """Clip and renormalize a float pdf to satisfy arithmetic_coder.py's needs.

    Two hard constraints from _CoderBase._get_intervals:
      1. every symbol needs probability >= Coder.p_min(base, precision), or
         quantization can round it to a zero-width interval and the coder raises;
      2. sum(pdf) must stay strictly below 1.0 after quantization, or the
         cumulative sum can exceed the coder's integer range.
    """
    p_min = 2.0 * base ** -(precision - 2)
    probs = np.clip(probs.astype(np.float64), p_min, None)
    probs /= probs.sum()
    # small safety margin below 1.0 to absorb float rounding in the coder's
    # integer cumsum step
    probs *= 1.0 - 1e-9
    return probs
"""
Compress a text file losslessly using a language model's own next-token
predictions as the probability source for arithmetic coding.

    uv run compress.py input.txt output.lmac \
        --checkpoint HuggingFaceTB/SmolLM2-135M

The core loop is the whole idea in five lines: ask the model what it thinks
comes next, hand that distribution plus the *actual* next token to the
arithmetic coder, then tell the model what really happened so it can predict
the token after that. Repeat once per token in the file.
"""

from __future__ import annotations

import argparse
import os
import time

import arithmetic_coder
from bitio import BitWriter
from container import write_container
from model import TokenPredictor


def compress(
    input_path: str,
    output_path: str,
    checkpoint: str,
    base: int = 2,
    precision: int = 32,
) -> None:
    with open(input_path, "rb") as f:
        original_bytes = f.read()
    text = original_bytes.decode("utf-8")

    print(f"Loading {checkpoint} ...")
    predictor = TokenPredictor(checkpoint)

    # add_special_tokens=False: we prime context ourselves via TokenPredictor's
    # start token in reset(); we don't want the tokenizer sneaking its own
    # special tokens into the content stream, which decode() would then need
    # to reconstruct token-for-token.
    token_ids = predictor.tokenizer(text, add_special_tokens=False)["input_ids"]

    # Sanity check *before* claiming losslessness, not after: some tokenizers
    # normalize whitespace or unicode on decode in ways that aren't perfectly
    # reversible. If this fails, this scheme silently would NOT round-trip --
    # better to catch it here than to discover it in decompress.py.
    roundtrip = predictor.tokenizer.decode(
        token_ids, clean_up_tokenization_spaces=False
    )
    if roundtrip != text:
        raise ValueError(
            "Tokenizer round-trip is not exact for this input -- "
            "token-level arithmetic coding cannot guarantee losslessness here. "
            "(First divergence is the thing to inspect.)"
        )

    max_ctx = predictor.max_context_length
    chunk_size = max_ctx - 1  # leave room for the BOS-primed first step
    num_chunks = max(1, -(-len(token_ids) // chunk_size))  # ceil div
    if num_chunks > 1:
        print(
            f"Input is {len(token_ids)} tokens, exceeds context length "
            f"{max_ctx}; splitting into {num_chunks} independently primed "
            f"chunks (each loses cross-chunk context, same limitation the "
            f"DeepMind paper hits with fixed-context models)."
        )

    writer = BitWriter()
    encoder = arithmetic_coder.Encoder(
        base=base, precision=precision, output_fn=writer.write_bit
    )

    t0 = time.perf_counter()
    predictor.reset()
    tokens_since_reset = 0
    for i, token_id in enumerate(token_ids):
        if tokens_since_reset >= chunk_size:
            predictor.reset()
            tokens_since_reset = 0

        pdf = predictor.next_token_pdf(base, precision)
        encoder.encode(pdf, token_id)
        predictor.advance(token_id)
        tokens_since_reset += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(token_ids)} tokens", end="\r")

    encoder.terminate()
    elapsed = time.perf_counter() - t0
    payload = writer.getvalue()

    meta = {
        "checkpoint": checkpoint,
        "base": base,
        "precision": precision,
        "num_tokens": len(token_ids),
        "chunk_size": chunk_size,
        "original_bytes": len(original_bytes),
    }
    write_container(output_path, meta, payload)

    # --- Reporting ---------------------------------------------------
    original_size = len(original_bytes)
    compressed_size = os.path.getsize(output_path)
    model_size = predictor.model_size_bytes()

    raw_ratio = compressed_size / original_size
    adjusted_ratio = (compressed_size + model_size) / original_size

    print()
    print(f"Original size:     {original_size:,} bytes")
    print(f"Compressed size:   {compressed_size:,} bytes")
    print(f"  raw ratio:        {raw_ratio:.4f}")
    print(f"Model on disk:     {model_size:,} bytes "
          f"({model_size / 1e6:.1f} MB) -- NOT included above")
    print(f"  adjusted ratio:   {adjusted_ratio:.4f}  "
          f"(if you count the model as part of the compressed payload, "
          f"as the paper's 'adjusted compression rate' does)")
    print(f"Time:              {elapsed:.1f}s "
          f"({len(token_ids) / elapsed:.1f} tokens/s)")
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument(
        "--checkpoint", default="HuggingFaceTB/SmolLM2-135M",
        help="Any causal LM on the Hugging Face Hub, base (non-instruct) "
             "checkpoints recommended.",
    )
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--precision", type=int, default=32)
    args = parser.parse_args()
    compress(
        args.input_path, args.output_path, args.checkpoint,
        base=args.base, precision=args.precision,
    )


if __name__ == "__main__":
    main()
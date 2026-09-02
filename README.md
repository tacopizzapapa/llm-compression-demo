# compress-demo

A working demonstration that language modeling *is* compression, per [Delétang et al., "Language Modeling Is Compression"](https://arxiv.org/abs/2309.10668). A causal LM's next-token predictions become the probability source for arithmetic coding. Better predictions, smaller output. That's the whole idea, and this repo just runs it.

Entirely Claude-generated. Not optimized, not scrutinized, not hardened. It's a toy you can run on your own hardware to watch the theory hold up.

## How it works

Arithmetic coding needs a probability distribution over "what comes next" at every step. Normally that comes from symbol frequencies. Here it comes from an LLM:

1. Ask the model what it thinks the next token is (a full distribution, not just the argmax).
2. Hand that distribution plus the *actual* next token to the arithmetic coder.
3. Tell the model what really happened, advance its KV cache, repeat.

The better the model's predictions, the fewer bits the coder needs to spend on the token that actually showed up. That's compression. Swap in a bigger model and watch the ratio improve — up to the point where the model itself gets too large to count as a fair trade.

`compress.py` is the whole loop, five lines at its core. `model.py` wraps a HuggingFace checkpoint (`TokenPredictor`) to expose exactly two operations: `next_token_pdf()` (look, don't advance) and `advance(token_id)` (commit and move state forward). `arithmetic_coder.py` is the actual encoder/decoder — lifted wholesale from [DeepMind's reference implementation](https://github.com/google-deepmind/language_modeling_is_compression). `bitio.py` packs the coder's bit stream into bytes. `container.py` is a trivial length-prefixed-JSON-header + payload format so the compressed file carries its own metadata (checkpoint, base, precision, token count).

Everything runs on CPU by design. GPU matmul isn't guaranteed bit-reproducible run to run, and bit-reproducibility is the entire game here — the same model given the same context must produce the exact same distribution every time, or the encode and (eventual) decode passes diverge.

## Running it

Needs [uv](https://docs.astral.sh/uv/). Python 3.11.

**1. Get something to compress.** A Gutenberg fetch script strips the license boilerplate for you:

```
uv run fetch_gutenberg.py https://www.gutenberg.org/cache/epub/100/pg100.txt shakespeare_small.txt 20000
```

The trailing number is an optional character cutoff — useful for a fast sanity-check run before committing to the full text.

**2. Compress it:**

```
uv run compress.py input.txt output.lmac --checkpoint HuggingFaceTB/SmolLM2-135M
```

`--checkpoint` takes any causal LM on the Hugging Face Hub; base (non-instruct) checkpoints are recommended. Defaults to `HuggingFaceTB/SmolLM2-135M`. Reports original size, compressed size, and the *adjusted* size — compressed payload plus the on-disk weight of the model itself, since the paper's honest accounting treats the model as part of what you'd have to ship to reproduce the result.

## Notes worth knowing before you run this

- **Decompression isn't implemented yet.** `output.lmac` is a one-way trip for now.
- **Lossless is enforced, not assumed.** Before compressing, the script round-trips the tokenizer's encode/decode on the full input and refuses to proceed if it doesn't match exactly — some tokenizers normalize whitespace or unicode in ways that quietly break reversibility.
- **Long inputs get chunked** at the model's max context length, each chunk independently primed. This throws away cross-chunk context — same limitation the DeepMind paper hits with any fixed-context model.
- **Bigger models compress better, until they don't.** The adjusted ratio (payload + model weights) is where that tradeoff becomes visible — a giant model can shrink the payload to nothing while being far larger than the original file itself.

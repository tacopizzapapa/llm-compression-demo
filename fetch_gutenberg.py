"""
Download a Project Gutenberg text, strip the legal boilerplate Gutenberg
wraps around every ebook, optionally truncate to a character count, and
write the result to a file -- ready to feed straight into compress.py.

    uv run fetch_gutenberg.py https://www.gutenberg.org/cache/epub/100/pg100.txt shakespeare.txt
    uv run fetch_gutenberg.py https://www.gutenberg.org/cache/epub/100/pg100.txt shakespeare_small.txt 20000
"""

from __future__ import annotations

import argparse
import re
import urllib.request

# Modern Gutenberg texts (the vast majority) bracket the actual content with
# lines like:
#   *** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
#   ...content...
#   *** END OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Some older Gutenberg texts are Latin-1.
        return raw.decode("latin-1")


def strip_boilerplate(text: str) -> str:
    start_match = _START_RE.search(text)
    end_match = _END_RE.search(text)

    if start_match is None or end_match is None:
        print(
            "Warning: couldn't find standard Gutenberg START/END markers -- "
            "leaving text as-is. Check the file for leftover license text."
        )
        return text

    content = text[start_match.end():end_match.start()]
    return content.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Direct link to a Gutenberg .txt file")
    parser.add_argument("output_path")
    parser.add_argument(
        "chars", nargs="?", type=int, default=None,
        help="Optional: truncate to this many characters after stripping "
             "boilerplate (grabs the first N chars, useful for a fast "
             "sanity-check run before compressing the whole thing).",
    )
    args = parser.parse_args()

    print(f"Downloading {args.url} ...")
    raw_text = fetch(args.url)
    print(f"  downloaded {len(raw_text):,} characters")

    stripped = strip_boilerplate(raw_text)
    print(f"  {len(stripped):,} characters after stripping boilerplate")

    if args.chars is not None:
        stripped = stripped[:args.chars]
        print(f"  truncated to {len(stripped):,} characters")

    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(stripped)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()

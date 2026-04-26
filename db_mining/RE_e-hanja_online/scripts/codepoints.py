"""
Codepoint enumeration for e-hanja online crawl.

List of Unicode ranges e-hanja advertises in their coverage page:
  Unified Han, Compat, Compat Supp, Kangxi Radicals, CJK Radicals Supplement,
  Ext A, Ext B, Ext C, Ext D. (E / F excluded by e-hanja.)

Gives ~76k codepoints. About 71,716 of them resolve on e-hanja; rest return 404.
"""

from __future__ import annotations

from typing import Iterable


BLOCKS: list[tuple[str, int, int]] = [
    # (name, start, end_inclusive)
    ("CJK Radicals Supplement", 0x2E80, 0x2EFF),     # 115 slots
    ("Kangxi Radicals",         0x2F00, 0x2FDF),     # 214 slots (ehanja covers 21)
    ("CJK Unified",             0x4E00, 0x9FFF),     # 20,989 slots (ehanja 20,950)
    ("CJK Compat",              0xF900, 0xFAFF),     # 472 slots
    ("CJK Ext A",               0x3400, 0x4DBF),     # 6,592 slots (ehanja 6,582)
    ("CJK Ext B",               0x20000, 0x2A6DF),   # 42,718 slots (ehanja 42,711)
    ("CJK Ext C",               0x2A700, 0x2B73F),   # 4,149 slots (ehanja 376)
    ("CJK Ext D",               0x2B740, 0x2B81F),   # 222 slots (ehanja 62)
    ("CJK Compat Supp",         0x2F800, 0x2FA1F),   # 542 slots
]


def enumerate_codepoints() -> Iterable[int]:
    """Yield every codepoint in e-hanja's advertised coverage blocks, in order."""
    for _name, start, end in BLOCKS:
        for cp in range(start, end + 1):
            yield cp


def total_count() -> int:
    return sum((end - start + 1) for _n, start, end in BLOCKS)


if __name__ == "__main__":
    print(f"total enumerated codepoints: {total_count()}")
    for name, start, end in BLOCKS:
        print(f"  {name:30s}  U+{start:04X}..U+{end:04X}  ({end - start + 1:,} slots)")

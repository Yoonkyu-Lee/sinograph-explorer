"""
System-font renderer for a single character.
Walks C:/Windows/Fonts, keeps only TTF/TTC/OTF that contain a glyph for the
target character, and renders one 256x256 PNG per (font file, face index).
"""

from __future__ import annotations

import argparse
import io
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path("C:/Windows/Fonts")
IMG_SIZE = 256
FONT_PT = 200
BG = 255
FG = 0


def iter_font_faces(path: Path):
    """Yield (face_index, family_name, subfamily_name, cmap_has_char_fn) for each face."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".ttc", ".otc"):
            coll = TTCollection(str(path))
            for idx, font in enumerate(coll.fonts):
                yield idx, font
        elif suffix in (".ttf", ".otf"):
            font = TTFont(str(path), lazy=True)
            yield 0, font
    except Exception as exc:
        print(f"[skip] {path.name}: {exc}", file=sys.stderr)


def face_name(font: TTFont) -> tuple[str, str]:
    name_table = font["name"]
    family = name_table.getBestFamilyName() or "Unknown"
    subfamily = name_table.getBestSubFamilyName() or ""
    return family, subfamily


def has_glyph(font: TTFont, char: str) -> bool:
    cp = ord(char)
    for table in font["cmap"].tables:
        if cp in table.cmap:
            return True
    return False


def render_one(font_path: Path, face_index: int, char: str, out_path: Path) -> bool:
    try:
        pil_font = ImageFont.truetype(str(font_path), FONT_PT, index=face_index)
    except Exception as exc:
        print(f"[skip] PIL cannot load {font_path.name}#{face_index}: {exc}", file=sys.stderr)
        return False

    img = Image.new("L", (IMG_SIZE, IMG_SIZE), BG)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=pil_font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (IMG_SIZE - w) // 2 - bbox[0]
    y = (IMG_SIZE - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=FG, font=pil_font)
    img.save(out_path)
    return True


def safe_tag(family: str, subfamily: str) -> str:
    base = f"{family}_{subfamily}".strip("_ ")
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("char", help="single character to render")
    parser.add_argument("--out", default=None, help="output directory (default: out/<char>)")
    parser.add_argument("--fonts-dir", default=str(FONTS_DIR))
    args = parser.parse_args()

    if len(args.char) != 1:
        raise SystemExit(f"expected single character, got {args.char!r}")
    char = args.char
    cp = f"U+{ord(char):04X}"

    script_dir = Path(__file__).resolve().parent
    out_root = Path(args.out) if args.out else script_dir.parent / "out" / f"{cp}_systemfonts"
    out_root.mkdir(parents=True, exist_ok=True)

    fonts_dir = Path(args.fonts_dir)
    candidates = sorted(
        p for p in fonts_dir.iterdir()
        if p.suffix.lower() in (".ttf", ".ttc", ".otf", ".otc")
    )

    print(f"char={char} ({cp}) {unicodedata.name(char, '?')}")
    print(f"scanning {len(candidates)} font files in {fonts_dir}")
    print(f"out: {out_root}")

    rendered = 0
    covered_faces = 0
    total_faces = 0

    for font_path in candidates:
        for face_index, tt in iter_font_faces(font_path):
            total_faces += 1
            if not has_glyph(tt, char):
                continue
            family, subfamily = face_name(tt)
            covered_faces += 1
            tag = safe_tag(family, subfamily)
            out_name = f"{font_path.stem}__{face_index}__{tag}.png"
            out_path = out_root / out_name
            if render_one(font_path, face_index, char, out_path):
                rendered += 1
                print(f"  ok  {font_path.name}#{face_index} [{family} / {subfamily}]")

    print(
        f"\nsummary: faces_scanned={total_faces} glyph_covered={covered_faces} rendered={rendered}"
    )


if __name__ == "__main__":
    main()

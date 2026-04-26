"""
Base glyph-mask sources for the synthetic pipeline.

A BaseSource produces a grayscale mask (L-mode, white=glyph on black bg) for
a given character literal. The mask is the 'raw glyph shape', independent of
any stylistic choices. It is the input to downstream effect stacks.

v1 implementations:
  - FontSource: rasterize from a TTF/TTC/OTF face

Planned later:
  - SvgStrokeSource: render from MakeMeAHanzi SVG stroke paths
  - ProceduralSource: shape-construction from decomposition + stroke primitives
  - HandwritingSource: sampled from a learned stroke distribution
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont


CANVAS_DEFAULT = 384    # work canvas, larger than output so effects don't clip
PAD_DEFAULT = 48        # margin so shadows/outlines survive the later crop


class BaseSource(ABC):
    """Produces a grayscale glyph mask for a literal.

    rng is passed in so stochastic sources (e.g. SvgStrokeSource with per-stroke
    variation) can vary per sample. Deterministic sources ignore it.
    """

    @abstractmethod
    def render_mask(self, char: str, rng=None) -> Image.Image | None:
        """Return L-mode mask of size (canvas, canvas). None if render fails."""

    @abstractmethod
    def tag(self) -> str:
        """Short filename-safe identifier for this source instance."""


@dataclass
class FontSource(BaseSource):
    font_path: Path
    face_index: int = 0
    family: str = ""
    subfamily: str = ""
    canvas: int = CANVAS_DEFAULT
    pad: int = PAD_DEFAULT
    # Canonical source-kind (for source-aware augment gating). Fixed for font.
    kind: str = "font"

    def render_mask(self, char: str, rng=None) -> Image.Image | None:
        # rng unused — font rasterization is deterministic per (face, char)
        target = self.canvas - 2 * self.pad
        lo, hi = 16, self.canvas
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                font = ImageFont.truetype(str(self.font_path), mid, index=self.face_index)
            except Exception:
                return None
            bbox = font.getbbox(char)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w <= target and h <= target:
                best = (font, bbox)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            try:
                font = ImageFont.truetype(str(self.font_path), 64, index=self.face_index)
            except Exception:
                return None
            best = (font, font.getbbox(char))
        font, bbox = best
        mask = Image.new("L", (self.canvas, self.canvas), 0)
        draw = ImageDraw.Draw(mask)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (self.canvas - w) // 2 - bbox[0]
        y = (self.canvas - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=255, font=font)
        return mask

    def tag(self) -> str:
        return f"{self.font_path.stem}-{self.face_index}"


def _has_glyph(tt: TTFont, char: str) -> bool:
    cp = ord(char)
    for table in tt["cmap"].tables:
        if cp in table.cmap:
            return True
    return False


def _iter_faces(path: Path):
    """Yield (face_index, TTFont) pairs."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".ttc", ".otc"):
            coll = TTCollection(str(path))
            for idx, tt in enumerate(coll.fonts):
                yield idx, tt
        elif suffix in (".ttf", ".otf"):
            tt = TTFont(str(path), lazy=True)
            yield 0, tt
    except Exception as exc:
        print(f"[skip] {path.name}: {exc}", file=sys.stderr)


# Cache of full font directory scan: (fonts_dir, canvas, pad) -> list of
# (FontSource, supported_codepoint_set). Filled on first call; subsequent
# calls just filter by cmap (sub-ms). Essential for corpus-scale generation
# where `discover_font_sources` is invoked per character (thousands of times).
_FONT_SCAN_CACHE: dict = {}


def _scan_all_faces(fonts_dir: Path, canvas: int, pad: int):
    key = (str(fonts_dir), canvas, pad)
    if key in _FONT_SCAN_CACHE:
        return _FONT_SCAN_CACHE[key]
    faces: list[tuple[FontSource, set[int]]] = []
    candidates = sorted(
        p for p in fonts_dir.iterdir()
        if p.suffix.lower() in {".ttf", ".ttc", ".otf", ".otc"}
    )
    for path in candidates:
        for idx, tt in _iter_faces(path):
            cps: set[int] = set()
            try:
                for table in tt["cmap"].tables:
                    cps.update(table.cmap.keys())
            except Exception:
                pass
            name = tt["name"]
            family = name.getBestFamilyName() or ""
            subfamily = name.getBestSubFamilyName() or ""
            fs = FontSource(
                font_path=path,
                face_index=idx,
                family=family,
                subfamily=subfamily,
                canvas=canvas,
                pad=pad,
            )
            faces.append((fs, cps))
    _FONT_SCAN_CACHE[key] = faces
    return faces


def discover_font_sources(
    fonts_dir: Path,
    char_filter: str | None = None,
    canvas: int = CANVAS_DEFAULT,
    pad: int = PAD_DEFAULT,
) -> list[FontSource]:
    """Walk fonts_dir, return FontSource for each face.

    If char_filter is given, only keep faces whose cmap covers that character.
    First call for a given (dir, canvas, pad) triple does the full scan;
    subsequent calls are O(#faces) set-membership checks.
    """
    all_faces = _scan_all_faces(fonts_dir, canvas, pad)
    if not char_filter:
        return [fs for fs, _ in all_faces]
    cp = ord(char_filter)
    return [fs for fs, cps in all_faces if cp in cps]

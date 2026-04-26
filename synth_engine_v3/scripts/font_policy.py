"""Font policy for synth_engine_v3 — v3-only wrappers around v2 font
discovery. Does three things:

  1. **External font dir merge** — scans `db_src/fonts/external/` in addition
     to the Windows system font dir, so serif / calligraphy / Nanum faces we
     download land in the same source pool.
  2. **Blacklist** — filenames (music / symbol / historic-only faces) that
     shouldn't render CJK chars are excluded by name match, even if their
     cmap accidentally claims coverage.
  3. **Tofu filter** — per `(font_face, char)` rendering check: if the mask
     has < threshold density OR < threshold bbox area we treat the result as
     a .notdef glyph and drop the face from the char's source list. The
     decision is cached per worker in an LRU, so corpus-scale generation pays
     the check only once per pair.

v2 files remain untouched. Callers replace direct v2 `resolve_base_sources`
calls with `get_font_sources_with_policy`.

Added 2026-04-23 (Codex font-diversity follow-up, Phase 83).
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

_V2_SCRIPTS = Path(__file__).resolve().parents[2] / "synth_engine_v2" / "scripts"
if str(_V2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_V2_SCRIPTS))

import base_source as v2_base_source  # noqa: E402
from fontTools.ttLib import TTCollection, TTFont  # noqa: E402

# External font dir. Combined home for auto-downloaded fonts + user-dropped
# bundles (Google Fonts CJK, HK/TC serif, Japanese brush, etc.). Scanned
# recursively so any future nested `static/` subfolders are included.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_FONTS_DIR = REPO_ROOT / "db_src" / "fonts" / "external"
EXTERNAL_FONTS_DIRS: list[Path] = [EXTERNAL_FONTS_DIR]

# Blacklist: match anywhere in the font path stem (case-insensitive). These
# are either non-CJK by design (music, math, historic) or cosmetic/icon fonts
# whose cmap may include accidental CJK entries mapping to .notdef.
_BLACKLIST_PATTERNS = [
    r"bravura",            # music notation
    r"phagspa",            # Mongolian script (not CJK despite shape overlap)
    r"wingding",
    r"webding",
    r"marlett",            # Windows UI glyphs
    r"mtextra",
    r"cambria\s*math",
    r"segoe\s*(emoji|mdl|icon|print|script)",
    r"ms\s*reference",
    r"symbol",
    r"holomdl2",           # Windows icon font
    r"sylfaen",            # non-CJK script coverage
    r"euro\s*sign",
]
_BLACKLIST_RE = re.compile(
    "|".join(f"(?:{p})" for p in _BLACKLIST_PATTERNS), re.IGNORECASE
)

# Subfamily patterns for "heavy" faces — Black / ExtraBold / Heavy / UltraBold.
# These render very thick strokes; combined with a dense high-stroke glyph the
# result is an unreadable blob. Excluded at source-resolution time when
# total_strokes ≥ _HEAVY_FILTER_STROKE_THRESHOLD.
_HEAVY_SUBFAMILY_RE = re.compile(
    r"\b(black|heavy|extra\s*bold|ultra\s*bold|extrablack|ultrablack)\b",
    re.IGNORECASE,
)
_HEAVY_FILTER_STROKE_THRESHOLD = 25

# Italic / oblique faces are unrealistic for printed/engraved CJK signage
# (sign / 현판 / 문화재 domain) and add distortion on top of existing
# perspective augment. Always excluded.
_ITALIC_SUBFAMILY_RE = re.compile(r"\b(italic|oblique)\b", re.IGNORECASE)


def _is_blacklisted(font_path: Path) -> bool:
    return bool(_BLACKLIST_RE.search(str(font_path)))


# Tofu thresholds — tuned to what a real hanzi mask looks like at 384×384.
# Real glyphs: 5k-50k lit pixels (3-35% of 147k). .notdef (small empty
# rectangle): < 800 lit pixels. Pick 0.3% of canvas as the cutoff.
_TOFU_PIXEL_RATIO = 0.003          # 0.3%
_TOFU_BBOX_RATIO = 0.03            # 3% of canvas edge

# Cache size: 20k (font, char) pairs comfortably per worker.
@lru_cache(maxsize=20000)
def _is_tofu_cached(font_path_str: str, face_index: int, char: str) -> bool:
    # Rebuild FontSource on the fly — the cheap path is cmap-filtered already.
    src = v2_base_source.FontSource(
        font_path=Path(font_path_str),
        face_index=face_index,
        family="", subfamily="",
        canvas=v2_base_source.CANVAS_DEFAULT,
        pad=v2_base_source.PAD_DEFAULT,
    )
    mask = src.render_mask(char)
    if mask is None:
        return True
    arr = np.asarray(mask)
    n_lit = int((arr > 127).sum())
    if n_lit < _TOFU_PIXEL_RATIO * arr.size:
        return True
    # bbox area check — a tiny filled rectangle passes the pixel count if
    # the cmap gave a notdef shape. Make sure the bbox is sensible.
    ys, xs = np.where(arr > 127)
    if ys.size == 0:
        return True
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    edge = mask.size[0]
    if bbox_w < _TOFU_BBOX_RATIO * edge or bbox_h < _TOFU_BBOX_RATIO * edge:
        return True
    return False


_EXTERNAL_SCAN_CACHE: dict = {}


def _should_skip_variable_font(path: Path) -> bool:
    """If this font is a *VariableFont_wght* file and a sibling `static/`
    dir has individual-weight TTFs of the same family, prefer the static
    weights (more face variety) and skip the VF file to avoid a duplicate
    default-weight render."""
    stem = path.stem
    if "VariableFont" not in stem and not stem.endswith("-VF"):
        return False
    static_dir = path.parent / "static"
    if not static_dir.is_dir():
        return False
    family_prefix = stem.split("-VariableFont")[0].split("-VF")[0]
    for sib in static_dir.iterdir():
        if sib.suffix.lower() in (".ttf", ".otf") and sib.stem.startswith(family_prefix):
            return True
    return False


def _iter_external_font_files(fonts_dir: Path):
    """Walk subtree, yield .ttf/.otf/.ttc/.otc paths with VF deduplication."""
    if not fonts_dir.exists():
        return
    exts = {".ttf", ".otf", ".ttc", ".otc"}
    for p in sorted(fonts_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            if _should_skip_variable_font(p):
                continue
            yield p


def _scan_external_dir(fonts_dir: Path, canvas: int, pad: int):
    """Recursive font scan for external dirs. Mirrors v2 `_scan_all_faces`
    behavior (TT name + cmap extraction) but walks subtrees and applies VF
    dedup. Cached per (dir, canvas, pad) like v2."""
    key = (str(fonts_dir), canvas, pad)
    cached = _EXTERNAL_SCAN_CACHE.get(key)
    if cached is not None:
        return cached
    faces: list[tuple[v2_base_source.FontSource, set[int]]] = []
    for path in _iter_external_font_files(fonts_dir):
        try:
            if path.suffix.lower() in (".ttc", ".otc"):
                coll = TTCollection(str(path))
                face_list = [(i, tt) for i, tt in enumerate(coll.fonts)]
            else:
                tt = TTFont(str(path), lazy=True)
                face_list = [(0, tt)]
        except Exception:
            continue
        for idx, tt in face_list:
            cps: set[int] = set()
            try:
                for tbl in tt["cmap"].tables:
                    cps.update(tbl.cmap.keys())
            except Exception:
                pass
            try:
                name = tt["name"]
                family = name.getBestFamilyName() or ""
                subfamily = name.getBestSubFamilyName() or ""
            except Exception:
                family, subfamily = path.stem, ""
            fs = v2_base_source.FontSource(
                font_path=path, face_index=idx,
                family=family, subfamily=subfamily,
                canvas=canvas, pad=pad,
            )
            faces.append((fs, cps))
    _EXTERNAL_SCAN_CACHE[key] = faces
    return faces


def _scan_dir_for_char(fonts_dir: Path, char: str) -> list[v2_base_source.FontSource]:
    """Return FontSources whose cmap covers `char`. Uses v2's flat scanner
    for the Windows system dir, and a recursive scanner for external dirs
    (handles Google Fonts style nested `static/` folders).
    """
    if not fonts_dir.exists():
        return []
    # External dirs are under db_src/fonts/ — use recursive scan
    try:
        is_external = any(
            fonts_dir.resolve() == d.resolve() for d in EXTERNAL_FONTS_DIRS
        )
    except Exception:
        is_external = False
    if is_external:
        cp = ord(char)
        scanned = _scan_external_dir(
            fonts_dir,
            v2_base_source.CANVAS_DEFAULT,
            v2_base_source.PAD_DEFAULT,
        )
        return [fs for (fs, cps) in scanned if cp in cps]
    # Windows system dir — v2's cached flat scan
    return v2_base_source.discover_font_sources(fonts_dir, char_filter=char)


def get_font_sources_with_policy(
    char: str,
    system_fonts_dir: Path = Path("C:/Windows/Fonts"),
    external_fonts_dir: Path | None = None,
    extra_external_dirs: list[Path] | None = None,
    filter_spec: str = "all",
    drop_tofu: bool = True,
    total_strokes: int = 0,
) -> list[v2_base_source.FontSource]:
    """Return FontSource list for `char` across system + external dirs, with
    blacklist removed and (optionally) tofu-producing faces dropped.

    `extra_external_dirs` supplements `external_fonts_dir`. When both params
    are left at default the scanner walks every dir in
    `EXTERNAL_FONTS_DIRS` (external/).

    `total_strokes` (≥ 0) triggers an extra filter: when ≥
    `_HEAVY_FILTER_STROKE_THRESHOLD` (currently 25) we drop Black / Heavy /
    ExtraBold / UltraBold subfamilies. Thick stroke weights on dense glyphs
    merge into an unreadable blob.
    """
    dirs: list[Path]
    if external_fonts_dir is None and extra_external_dirs is None:
        dirs = list(EXTERNAL_FONTS_DIRS)
    else:
        dirs = []
        if external_fonts_dir is not None:
            dirs.append(external_fonts_dir)
        if extra_external_dirs:
            dirs.extend(extra_external_dirs)

    sources: list[v2_base_source.FontSource] = []
    sources.extend(_scan_dir_for_char(system_fonts_dir, char))
    for d in dirs:
        sources.extend(_scan_dir_for_char(d, char))

    # Optional filter_spec (v2 resolve_base_sources uses `filter`: "all" or
    # comma-separated substrings against font path stem).
    if filter_spec and filter_spec != "all":
        wanted = [w.strip().lower() for w in str(filter_spec).split(",") if w.strip()]
        sources = [s for s in sources
                   if any(w in s.font_path.stem.lower() for w in wanted)]

    # Blacklist by filename
    sources = [s for s in sources if not _is_blacklisted(s.font_path)]

    # Italic / oblique faces — always excluded for CJK.
    sources = [
        s for s in sources
        if not _ITALIC_SUBFAMILY_RE.search(getattr(s, "subfamily", "") or "")
    ]

    # Heavy-face filter for high-stroke chars
    if total_strokes >= _HEAVY_FILTER_STROKE_THRESHOLD:
        sources = [
            s for s in sources
            if not _HEAVY_SUBFAMILY_RE.search(getattr(s, "subfamily", "") or "")
        ]

    # Tofu filter
    if drop_tofu:
        kept = []
        for s in sources:
            if _is_tofu_cached(str(s.font_path), s.face_index, char):
                continue
            kept.append(s)
        sources = kept

    return sources

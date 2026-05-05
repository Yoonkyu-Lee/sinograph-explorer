"""Pluggable Commodity OCR adapter pattern for demo Stage 1 bench.

Design goal: a uniform `OCRAdapter` interface so the bench script can loop
`for adapter in adapters: chars, lat = adapter.recognize_topk(img, k=5)`
and add new commodity OCRs (PaddleOCR, Google Lens, …) by just implementing
one class — no bench-script changes.

Top-k semantics — important honest caveat:
  Most commodity OCRs are *single-best-prediction* engines. Tesseract's PSM 10
  and EasyOCR's recognizer return one hypothesis per region. We expose
  `recognize_topk(img, k)` returning a list of length 1..k; engines that
  cannot give alternatives just return [top1] and the bench scores top-1
  and top-k accordingly (top-k will equal top-1 in those cases).

  v4 SCER's cosine-NN over a 98k anchor DB is *natively* a ranked
  retrieval, so top-5 is free — that contrast is itself a demo point.

Pre-processing for small isolated single chars:
  - Our test PNGs are ~40×40 px. Tesseract / EasyOCR recognizers expect
    ≥ ~30 px char height with clean white-bg composition. We RGBA→RGB-on-
    white composite and BICUBIC upscale 4× before passing in.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

from PIL import Image


class OCRAdapter:
    """Base class. Subclasses override `name`, `install_size_mb`, `recognize_topk`."""

    name: str = "base"
    install_size_mb: float = 0.0
    notes: str = ""

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        """Return ([top1, top2, ...], latency_ms) — list length 1..k.

        Engines that don't expose ranked alternatives return [top1].
        Empty prediction → [].
        """
        raise NotImplementedError

    # back-compat for any caller still using the old API
    def recognize(self, img_path: Path) -> Tuple[str, float]:
        chars, ms = self.recognize_topk(img_path, k=1)
        return (chars[0] if chars else ""), ms

    def warmup(self, img_path: Path, n: int = 1) -> None:
        for _ in range(n):
            self.recognize_topk(img_path, k=1)


# =============================================================================
# Shared preprocessing — RGBA→RGB on white + BICUBIC upscale
# =============================================================================

def _flatten_rgba_and_upscale(img_path: Path, scale: int = 4) -> Image.Image:
    img = Image.open(img_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if scale != 1:
        img = img.resize(
            (img.size[0] * scale, img.size[1] * scale), Image.BICUBIC
        )
    return img


# =============================================================================
# Tesseract — pytesseract wrapper. Returns single best (no native top-k).
# =============================================================================

class TesseractAdapter(OCRAdapter):
    """tesseract-ocr 5.x. Per-language instances expose how each commodity
    OCR is *language-specific* — the demo's universal-sinograph contrast.

    Top-k: NOT supported natively for PSM 10 single-char mode. Returns
    [top1] only. (Tesseract's ALTO XML alternatives at PSM 10 have proven
    empty in our environment; we don't fake topk via multi-PSM ensembles.)
    """

    install_size_mb = 50.0  # tesseract-ocr core; +12-15 MB per lang traineddata

    def __init__(self, lang: str = "jpn+chi_tra+chi_sim",
                 psm: int = 10, scale: int = 4):
        import pytesseract  # noqa: F401
        self._pt = pytesseract
        self.lang = lang
        self.psm = psm
        self.scale = scale
        # Friendly short label per row, e.g. "Tess[jpn]" or "Tess[multi]"
        if "+" in lang:
            self.name = f"Tess[multi]"
        else:
            self.name = f"Tess[{lang}]"
        self.notes = (f"lang={lang}, PSM=10 single char, 4× upscale; "
                      f"top-1 only (no native top-k)")

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        img = _flatten_rgba_and_upscale(img_path, scale=self.scale)
        t0 = time.perf_counter()
        raw = self._pt.image_to_string(
            img, lang=self.lang, config=f"--psm {self.psm}"
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        for ch in raw:
            if ch.strip():
                return [ch], dt_ms
        return [], dt_ms


# =============================================================================
# EasyOCR — neural CRNN. Uses Reader.recognize() to bypass the (small-image
# unfriendly) detection step and run recognizer-only on the whole patch.
# =============================================================================

class EasyOCRAdapter(OCRAdapter):
    """EasyOCR 1.7+. Recognizer-only mode: bypass detection, recognize whole
    image as one region.

    Per-language instance (EasyOCR's hard limit: one CJK family per Reader).
    Each instance is a separate ~70 MB model load — we instantiate one per
    language we want to bench, intentionally to show language-specific
    behaviour. v4 SCER serves all of these in a single 11 MB model.

    Top-k: greedy / beamSearch decoder both return only the best path
    through EasyOCR's public API. → Returns [top1].
    """

    install_size_mb = 70.0  # per-Reader recognition model (ja or ch_tra or ch_sim)

    def __init__(self, langs=("ja", "en"), beam_width: int = 1, scale: int = 4):
        # Silence the noisy PyTorch DataLoader pin_memory warning that fires
        # on every recognize() call when no GPU/accelerator is present.
        # (Pi has no CUDA — pin_memory has no effect, the warning is just
        # informational and floods stdout/stderr in a 20-image bench loop.)
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=r".*pin_memory.*no accelerator.*",
        )
        import numpy as np  # noqa: F401
        import easyocr
        self.reader = easyocr.Reader(list(langs), gpu=False, verbose=False)
        self.beam_width = beam_width
        self.scale = scale
        # Friendly label, e.g. "Easy[ja]" / "Easy[ch_tra]"
        primary = next((l for l in langs if l != "en"), langs[0])
        self.name = f"Easy[{primary}]"
        self.notes = (f"langs={'+'.join(langs)}, recognizer-only "
                      f"(bypass detect), {('beam=' + str(beam_width)) if beam_width > 1 else 'greedy'}; "
                      f"top-1 only")

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        import numpy as np
        img = _flatten_rgba_and_upscale(img_path, scale=self.scale)
        arr = np.array(img)
        H, W = arr.shape[:2]
        decoder = "beamsearch" if self.beam_width > 1 else "greedy"
        t0 = time.perf_counter()
        res = self.reader.recognize(
            arr, horizontal_list=[[0, W, 0, H]], free_list=[],
            decoder=decoder, beamWidth=self.beam_width,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if not res:
            return [], dt_ms
        text = res[0][1]
        for ch in text:
            if ch.strip():
                return [ch], dt_ms
        return [], dt_ms


# =============================================================================
# PaddleOCR — neural. Heavy install; included as a stub to enable later.
# =============================================================================

class PaddleOCRAdapter(OCRAdapter):
    """PaddleOCR. Strong CJK accuracy but heavy install (1-1.5 GB).

    KNOWN-LIMITATION on Raspberry Pi 5 (ARM64) + Python 3.13:
      - pypi wheel for `paddlepaddle` only ships 3.x for ARM64+Py3.13.
      - paddlepaddle 3.x consistently segfaults during the C++ predictor
        initialization on Pi 5 (libpaddle SaveOrLoadPirParameters → SIGSEGV).
      - paddlepaddle 2.x has no Python 3.13 wheel.
      → On Pi, this adapter is effectively unusable. Demo point: "even
        the strongest CJK commodity OCR has platform-portability gaps;
        v4 SCER ships as an 11 MB TFLite that runs anywhere with
        ai-edge-litert."

    On x86_64 dev machines, paddlepaddle 3.x usually works — the adapter
    is left functional for that path. Set env PADDLEOCR_FORCE=1 to attempt
    init on Pi anyway (will likely segfault).
    """

    install_size_mb = 1200.0

    def __init__(self, lang: str = "ch", scale: int = 4):
        import os
        import platform
        # Guard against the known Pi5 + Py3.13 segfault.
        is_arm64 = platform.machine() in ("aarch64", "arm64")
        if is_arm64 and not os.environ.get("PADDLEOCR_FORCE"):
            raise RuntimeError(
                "PaddleOCR is incompatible with Pi 5 (ARM64) + Python 3.13: "
                "paddlepaddle 3.x segfaults on init, paddlepaddle 2.x has "
                "no aarch64+py313 wheel. This is a commodity-OCR portability "
                "gap that v4 SCER's INT8 TFLite avoids. "
                "Set PADDLEOCR_FORCE=1 to attempt anyway."
            )
        from paddleocr import PaddleOCR
        # PaddleOCR 3.x API: predict() returns OcrResult objects.
        self._ocr = PaddleOCR(lang=lang)
        self.lang = lang
        self.scale = scale
        self.name = f"Paddle[{lang}]"
        self.notes = f"lang={lang} (PaddleOCR 3.x predict API), top-1 only"

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        import numpy as np
        img = _flatten_rgba_and_upscale(img_path, scale=self.scale)
        t0 = time.perf_counter()
        res = self._ocr.predict(np.array(img))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Result format varies between versions; conservatively try both.
        if not res:
            return [], dt_ms
        try:
            # 3.x OCRResult: list of dict-like, key "rec_texts" is a list of strings
            first = res[0]
            texts = first.get("rec_texts") or first.get("texts") or []
            if texts:
                txt = texts[0]
                for ch in txt:
                    if ch.strip():
                        return [ch], dt_ms
        except (AttributeError, KeyError, TypeError):
            pass
        # 2.x style fallback: list[list[(bbox, (text, conf))]]
        try:
            text = res[0][0][1][0]
            for ch in text:
                if ch.strip():
                    return [ch], dt_ms
        except (IndexError, TypeError):
            pass
        return [], dt_ms


# =============================================================================
# cnocr — Chinese-specific lightweight OCR (breezedeus). ONNX-based,
# uses RapidOCR PP-OCRv5 weights. Tiny + very fast on Pi CPU.
# =============================================================================

class CnocrAdapter(OCRAdapter):
    """cnocr 2.3+ — Chinese-specialist OCR (PP-OCRv5 ONNX weights via RapidOCR).

    Bypasses detection: ``ocr_for_single_line`` runs only the recognizer
    on the whole image. Fast (~10-30 ms/char on Pi CPU, ONNX runtime).

    Top-k: API returns one ``{text, score}`` per call. → [top1].
    """

    install_size_mb = 100.0  # cnocr + cnstd ONNX models

    def __init__(self, scale: int = 4):
        from cnocr import CnOcr
        self._ocr = CnOcr()
        self.scale = scale
        self.name = "cnocr"
        self.notes = ("PP-OCRv5 ONNX (RapidOCR backend), single-line "
                      "recognizer-only; top-1 only")

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        import numpy as np
        img = _flatten_rgba_and_upscale(img_path, scale=self.scale)
        arr = np.array(img)
        t0 = time.perf_counter()
        res = self._ocr.ocr_for_single_line(arr)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if not res or not isinstance(res, dict):
            return [], dt_ms
        text = res.get("text", "")
        for ch in text:
            if ch.strip():
                return [ch], dt_ms
        return [], dt_ms


# =============================================================================
# Manga-OCR — Hugging Face ViT encoder + GPT2 decoder, trained on Japanese
# manga panels. Specialist demo baseline: strong on Japanese (incl. kanji /
# vertical text), weak on Chinese-only or Korean-only glyphs.
# =============================================================================

class MangaOcrAdapter(OCRAdapter):
    """manga-ocr 0.1.x — Japanese-specialist transformer OCR.

    Model: kha-white/manga-ocr-base (~440 MB, downloaded to HF cache on
    first init). Encoder = ViT, decoder = GPT2-style autoregressive.

    Top-k: not exposed by the public ``MangaOcr`` callable (greedy decode
    only). Beam alternatives would require dropping into transformers'
    ``generate(num_beams=k, num_return_sequences=k, return_dict_in_generate=True)``
    on the underlying ``model``; we don't go there for the demo. → [top1].

    Latency: ~0.7-1.5 s/image on Pi 5 CPU (no GPU).
    """

    install_size_mb = 440.0  # HF model on disk

    def __init__(self, scale: int = 4):
        # Silence the same DataLoader pin_memory warning EasyOCR triggered.
        import warnings
        warnings.filterwarnings(
            "ignore", message=r".*pin_memory.*no accelerator.*",
        )
        from manga_ocr import MangaOcr
        self._mocr = MangaOcr()
        self.scale = scale
        self.name = "Manga"
        self.notes = (
            "kha-white/manga-ocr-base (ViT+GPT2), Japanese-specialist; "
            "top-1 only (greedy decode)"
        )

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        img = _flatten_rgba_and_upscale(img_path, scale=self.scale)
        t0 = time.perf_counter()
        text = self._mocr(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if not text:
            return [], dt_ms
        for ch in text:
            if ch.strip():
                return [ch], dt_ms
        return [], dt_ms


# =============================================================================
# Google Cloud Vision — REST OCR via API key. Cloud, no local model.
# =============================================================================

class GoogleVisionAdapter(OCRAdapter):
    """Google Cloud Vision OCR (TEXT_DETECTION) via REST + API key.

    Auth: set ``GOOGLE_VISION_API_KEY`` env var with your key.
    Get one at https://console.cloud.google.com/apis/credentials and
    enable Vision API at
    https://console.cloud.google.com/apis/library/vision.googleapis.com .

    Pricing: first 1000 OCR calls/month free, then $1.50/1000.
    Per-call latency 200-1000 ms (network + service).

    Top-k: native API returns multiple ``textAnnotations``, but for a
    single-region image the first annotation is the merged whole-text
    string and subsequent annotations are per-symbol breakdowns of that
    same string — not ranked alternatives. Returns [top1].
    """

    install_size_mb = 0.0  # cloud — no local model

    def __init__(self, scale: int = 4, language_hints=("zh", "ja", "ko")):
        import os
        self.api_key = os.environ.get("GOOGLE_VISION_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set GOOGLE_VISION_API_KEY env var. Get a key at "
                "https://console.cloud.google.com/apis/credentials, "
                "then enable Vision API for your project."
            )
        import requests  # noqa: F401
        self.endpoint = (
            "https://vision.googleapis.com/v1/images:annotate"
            f"?key={self.api_key}"
        )
        self.scale = scale
        self.language_hints = list(language_hints)
        self.name = "GVision"
        self.notes = (
            f"cloud REST (TEXT_DETECTION), hints={'+'.join(self.language_hints)}; "
            f"top-1 only"
        )

    def recognize_topk(self, img_path: Path, k: int = 5
                       ) -> Tuple[List[str], float]:
        import base64
        import io
        import requests

        img = _flatten_rgba_and_upscale(img_path, scale=self.scale)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        content = base64.b64encode(buf.getvalue()).decode("ascii")

        body = {
            "requests": [{
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION", "maxResults": k}],
                "imageContext": {"languageHints": self.language_hints},
            }]
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(self.endpoint, json=body, timeout=15)
        except requests.exceptions.RequestException:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return [], dt_ms
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if r.status_code != 200:
            # 403 = key invalid / API not enabled / quota exhausted
            return [], dt_ms
        data = r.json()
        try:
            resp = data["responses"][0]
            if "error" in resp:
                return [], dt_ms
            ta = resp.get("textAnnotations", [])
            if not ta:
                return [], dt_ms
            # First annotation = full detected text; pick its first non-ws char.
            text = ta[0].get("description", "")
            for ch in text:
                if ch.strip():
                    return [ch], dt_ms
        except (KeyError, IndexError, TypeError):
            pass
        return [], dt_ms


# =============================================================================
# Registry — bench script imports `ADAPTER_SPECS` and `ADAPTER_GROUPS`.
# Each spec is (class, kwargs); name is set per-instance from kwargs.
# =============================================================================

ADAPTER_SPECS = {
    "tesseract":          (TesseractAdapter, dict(lang="jpn+chi_tra+chi_sim")),
    "tesseract-jpn":      (TesseractAdapter, dict(lang="jpn")),
    "tesseract-chi_tra":  (TesseractAdapter, dict(lang="chi_tra")),
    "tesseract-chi_sim":  (TesseractAdapter, dict(lang="chi_sim")),
    "easyocr-ja":         (EasyOCRAdapter,   dict(langs=("ja", "en"))),
    "easyocr-ch_tra":     (EasyOCRAdapter,   dict(langs=("ch_tra", "en"))),
    "easyocr-ch_sim":     (EasyOCRAdapter,   dict(langs=("ch_sim", "en"))),
    "paddleocr-ch":       (PaddleOCRAdapter, dict(lang="ch")),
    "paddleocr-japan":    (PaddleOCRAdapter, dict(lang="japan")),
    "manga":              (MangaOcrAdapter,  dict()),
    "cnocr":              (CnocrAdapter,     dict()),
    "gvision":            (GoogleVisionAdapter, dict()),
}

ADAPTER_GROUPS = {
    # Default: Tesseract + 3 EasyOCR per-lang + PaddleOCR + Manga-OCR.
    # GVision excluded by default (requires GCP billing). Add via "cloud" or
    # explicit gvision id when API key + billing are set up.
    # Demo points across the row:
    #  - Tesseract: traditional ML, multi-lang traineddata
    #  - EasyOCR ×3: neural CRNN, language silos
    #  - PaddleOCR: strong CJK SDK, but Pi ARM64 incompat (graceful skip)
    #  - Manga-OCR: Japanese-specialist transformer (ViT + GPT2)
    "all":           ["tesseract", "easyocr-ja", "easyocr-ch_tra",
                      "easyocr-ch_sim", "paddleocr-ch", "cnocr", "manga"],
    "tesseract":     ["tesseract"],
    "tesseract-langs": ["tesseract-jpn", "tesseract-chi_tra", "tesseract-chi_sim"],
    "easyocr":       ["easyocr-ja", "easyocr-ch_tra", "easyocr-ch_sim"],
    "paddleocr":     ["paddleocr-ch", "paddleocr-japan"],
    "manga":         ["manga"],
    "cnocr":         ["cnocr"],
    "cloud":         ["gvision"],
    "offline":       ["tesseract", "easyocr-ja", "easyocr-ch_tra",
                      "easyocr-ch_sim", "cnocr", "manga"],
    "all-with-cloud":["tesseract", "easyocr-ja", "easyocr-ch_tra",
                      "easyocr-ch_sim", "paddleocr-ch", "cnocr", "manga",
                      "gvision"],
}


def resolve_adapter_spec(spec: str):
    """Take a string like 'all' or 'tesseract,easyocr-ja' and return list of
    (id, AdapterClass, kwargs) tuples. Unknown ids raise ValueError."""
    out = []
    seen = set()
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        ids = ADAPTER_GROUPS.get(token, [token])
        for aid in ids:
            if aid in seen:
                continue
            if aid not in ADAPTER_SPECS:
                raise ValueError(
                    f"unknown adapter id '{aid}'. "
                    f"Known: {sorted(ADAPTER_SPECS)}; "
                    f"groups: {sorted(ADAPTER_GROUPS)}"
                )
            cls, kwargs = ADAPTER_SPECS[aid]
            out.append((aid, cls, kwargs))
            seen.add(aid)
    return out


# back-compat alias kept so older callers work
DEFAULT_ADAPTERS = [TesseractAdapter, EasyOCRAdapter]


if __name__ == "__main__":
    # smoke
    import sys
    if len(sys.argv) < 2:
        print("usage: ocr_adapters.py <image.png> [<spec>]\n"
              "  spec is comma-separated ids/groups, e.g. 'all', "
              "'tesseract,easyocr-ja'")
        sys.exit(1)
    img = Path(sys.argv[1])
    spec = sys.argv[2].lower() if len(sys.argv) >= 3 else "all"
    for aid, cls, kwargs in resolve_adapter_spec(spec):
        a = cls(**kwargs)
        a.warmup(img, n=1)
        chars, ms = a.recognize_topk(img, k=5)
        print(f"[{aid:18s} {a.name:12s}] {img.name} → {chars} ({ms:.1f} ms)")

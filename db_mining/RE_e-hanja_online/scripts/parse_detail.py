"""Parse e-hanja_online detail HTML (jajun_contentA.asp response) → detail.jsonl.

Input:
  db_mining/RE_e-hanja_online/data/detail/{HEX}.html      (75,669 success)
  db_mining/RE_e-hanja_online/data/detail/{HEX}.404       (427 404 markers)

For each success file, extract the fields that are NOT already covered by
tree.jsonl (which carries getHunum + getJahae + getSchoolCom). Fields extracted:

  total_strokes, radical_strokes
  radical = { char, variant, name, etymology }
  classification = { education_level, hanja_grade, name_use }
  pinyin
  english
  shape = { representative: {char, gloss}, components: [{char, gloss}, ...] }
  etymology = { type, description }
  word_usage
  synonyms = [{char, gloss}, ...]
  related_words = [{word, reading, word_id}, ...]

Output:
  db_src/e-hanja_online/detail.jsonl            success records
  db_src/e-hanja_online/detail_404.jsonl        404 codepoints (audit)
  db_src/e-hanja_online/detail_anomalies.jsonl  parse warnings per record

Anomalies include missing-but-expected fields, unknown label text, etc.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


IN = Path("d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_mining/RE_e-hanja_online/data/detail")
OUT_DIR = Path("d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_src/e-hanja_online")
OUT_JSONL = OUT_DIR / "detail.jsonl"
OUT_404 = OUT_DIR / "detail_404.jsonl"
OUT_ANOM = OUT_DIR / "detail_anomalies.jsonl"


_TAG_LABEL_RE = re.compile(r"\s+")
_PINYIN_RE = re.compile(r"\〔\s*(.*?)\s*\〕")
_WORD_ID_RE = re.compile(r"pop_word\.asp\?word_id=(\d+)")
_WORD_USAGE_RE = re.compile(r"※\s*단어로\s*쓰일\s*때")

# Upper-level labels we actively extract (with 2 NBSP expansion).
# Everything else with `w3-light-grey` is a sub-label (e.g. "교육용" inside 분류)
# or a dropdown-hover tag (동자/유의/간체자/…) and should silently pass.
_SUPPRESSED_LABELS = {
    # sub-labels of 분류
    "교육용", "검정", "대법원",
    # dropdown-hover relation tags (already in tree.jsonl, or extracted by
    # _extract_detail_only_relations below)
    "동자", "유의", "간체자", "약자", "속자", "고자", "와자", "본자", "통자",
    "일본한자", "일본자", "상대", "별자",
    "본 자", "속 자", "약 자", "고 자", "와 자", "통 자",
}

# dropdown-hover tags that are NOT in tree.jsonl (schoolCom). Extracted into
# rec["related_characters"]. Translation:
#   유의 → synonyms (similar meaning)
#   상대 → opposites (opposing/contrasting)
#   별자 → alt_forms (alternative forms, distinct from the schoolCom typed variants)
_DETAIL_ONLY_RELATIONS = {"유의": "synonyms", "상대": "opposites", "별자": "alt_forms"}
# pinyin values that mean "unknown/absent"
_PINYIN_NULL = {"-", "*", "", "- (-)", "*(-)"}


def _norm(s: str) -> str:
    """Collapse whitespace + NBSP."""
    return _TAG_LABEL_RE.sub(" ", s.replace("\xa0", " ")).strip()


def _label_text(span) -> str:
    """Extract a label span's text, minus decorative spacing."""
    return _norm(span.get_text(" ", strip=True))


def _find_value_div(label_span):
    """Given a label <span class='w3-tag w3-round-large w3-light-grey ...'>, find
    the sibling `div.w3-rest` that carries the field's value (pattern used
    throughout the detail HTML).
    """
    row = label_span.find_parent("div", class_="w3-row")
    if row is None:
        return None
    return row.find("div", class_="w3-rest")


def _extract_int(text: str) -> int | None:
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else None


def _extract_stroke_count(val_div) -> tuple[int | None, int | None]:
    """획수 block: main count + (부수 획수: N)."""
    if val_div is None:
        return None, None
    text = val_div.get_text(" ", strip=True)
    total = None
    radical_strokes = None
    # total: first <a> or first number
    m = re.search(r"^\s*(\d+)", text)
    if m:
        total = int(m.group(1))
    m = re.search(r"부수\s*획수\s*[:\s]\s*(\d+)", text)
    if m:
        radical_strokes = int(m.group(1))
    return total, radical_strokes


def _extract_radical(val_div) -> dict | None:
    """부수 block: main radical char + optional simplified variant + name +
    optional etymology (in busuTxt div)."""
    if val_div is None:
        return None
    out: dict = {}
    # Main radical: first <a> with keyfield=busu has a <span>char</span>
    anchors = val_div.find_all("a")
    if anchors:
        first = anchors[0]
        if "keyfield=busu" in (first.get("href") or ""):
            span = first.find("span")
            if span:
                ch = span.get_text(strip=True)
                if ch:
                    out["char"] = ch
    # Simplified variant in a <sup>. May be comma-separated (e.g. "乛,乚,⺄").
    sup = val_div.find("sup")
    if sup:
        variant_span = sup.find("span")
        if variant_span:
            v = variant_span.get_text(strip=True)
            if v:
                parts = [p.strip() for p in v.split(",") if p.strip()]
                out["variant"] = parts if len(parts) > 1 else parts[0]
    # Radical name: second keyfield=busu anchor's inner span (e.g. "쇠금部")
    for a in anchors[1:]:
        if "keyfield=busu" in (a.get("href") or ""):
            spans = a.find_all("span")
            for sp in spans:
                t = sp.get_text(strip=True)
                if t and t not in ("(", ")"):
                    out["name"] = t
                    break
            break
    # Etymology hidden in busuTxt div
    busu_txt = val_div.find("div", id="busuTxt")
    if busu_txt:
        ety = _norm(busu_txt.get_text(" ", strip=True))
        if ety:
            out["etymology"] = ety
    return out or None


def _extract_classification(val_div) -> dict | None:
    """분류 block: alternating 태그 spans."""
    if val_div is None:
        return None
    out: dict = {}
    # Walk spans with class w3-border... (value) preceded by tag (label).
    spans = val_div.find_all("span")
    current_label = None
    for sp in spans:
        cls = sp.get("class") or []
        text = _norm(sp.get_text(" ", strip=True))
        if not text:
            continue
        if "w3-light-grey" in cls:
            current_label = text
        elif "w3-border" in cls and current_label:
            if current_label == "교육용":
                out["education_level"] = text
            elif current_label == "검정":
                out["hanja_grade"] = text
            elif current_label == "대법원":
                out["name_use"] = text
            current_label = None
    return out or None


def _extract_pinyin(val_div) -> str | None:
    if val_div is None:
        return None
    text = val_div.get_text(" ", strip=True)
    m = _PINYIN_RE.search(text)
    val = _norm(m.group(1)) if m else _norm(text)
    if not val or val in _PINYIN_NULL:
        return None
    return val


def _extract_english(val_div) -> str | None:
    if val_div is None:
        return None
    span = val_div.find("span", class_="eng")
    if span:
        return _norm(span.get_text(" ", strip=True))
    return _norm(val_div.get_text(" ", strip=True)) or None


def _extract_shape(val_div) -> dict | None:
    """모양자 block: representative with tooltip + components."""
    if val_div is None:
        return None
    items = []
    for a in val_div.find_all("a"):
        href = a.get("href") or ""
        if "keyfield=shape" not in href:
            continue
        # outer span is the clickable char; nested <span class='w3-text w3-tag...'>
        # is the tooltip gloss.
        outer_span = a.find("span", class_="w3-tooltip")
        if outer_span is None:
            continue
        tooltip = outer_span.find("span", class_="w3-tag")
        gloss = _norm(tooltip.get_text(" ", strip=True)) if tooltip else None
        # remove tooltip from outer to get just the char
        if tooltip:
            tooltip.extract()
        ch = _norm(outer_span.get_text(" ", strip=True))
        if ch:
            items.append({"char": ch, "gloss": gloss})
    if not items:
        return None
    return {"representative": items[0], "components": items[1:]}


def _extract_etymology(val_div) -> dict | None:
    """자원 block: optional bolded type anchor + description span."""
    if val_div is None:
        return None
    out: dict = {}
    # type anchor: <a href="...?keyfield=theory...">...(<span style="font-weight: bold;">형성문자</span>)...</a>
    for a in val_div.find_all("a"):
        href = a.get("href") or ""
        if "keyfield=theory" in href:
            bold = a.find("span", style=lambda s: s and "font-weight" in s)
            if bold:
                out["type"] = _norm(bold.get_text(" ", strip=True))
            break
    # description: last grey span
    gray_spans = [s for s in val_div.find_all("span") if (s.get("style") or "").find("color:gray") != -1]
    if gray_spans:
        desc = _norm(gray_spans[-1].get_text(" ", strip=True))
        if desc:
            out["description"] = desc
    return out or None


def _extract_word_usage(soup: BeautifulSoup) -> str | None:
    """※ 단어로 쓰일 때 block (no label tag — identify by text marker)."""
    marker = soup.find(string=_WORD_USAGE_RE)
    if marker is None:
        return None
    li = marker.find_parent("li")
    if li is None:
        return None
    # The meaning text is the large w3-text-blue span(s)
    spans = li.find_all("span", class_="w3-text-blue")
    if not spans:
        # fallback — grab the whole w3-rest
        rest = li.find("div", class_="w3-rest")
        if rest:
            return _norm(rest.get_text(" ", strip=True))
        return None
    return " ".join(_norm(s.get_text(" ", strip=True)) for s in spans).strip() or None


def _extract_detail_only_relations(soup: BeautifulSoup) -> dict:
    """dropdown-hover blocks with relation tags that are NOT already in
    tree.jsonl's schoolCom (i.e. 유의 / 상대 / 별자). Returns
    {relation_key: [{char, gloss}, ...]}. Relations not in the allowlist are
    silently ignored here (they're either in schoolCom or suppressed)."""
    out: dict[str, list[dict]] = {}
    for block in soup.find_all("div", class_="w3-dropdown-hover"):
        # label tag lives as a <span class="w3-tag ...">relation_name</span>
        tag_span = None
        for sp in block.find_all("span", class_="w3-tag"):
            # inside the main block body, not nested tooltip
            if sp.find_parent("div", class_="w3-dropdown-content"):
                continue
            tag_span = sp
            break
        if tag_span is None:
            continue
        rel = _norm(tag_span.get_text(" ", strip=True))
        key = _DETAIL_ONLY_RELATIONS.get(rel)
        if key is None:
            continue
        a = block.find("a")
        ch = None
        if a:
            onclick = a.get("href") or ""
            m = re.search(r"SetStorageA\('([^':]+):", onclick)
            if m:
                ch = m.group(1)
        gloss = None
        content = block.find("div", class_="w3-dropdown-content")
        if content:
            gloss_span = content.find("span", class_=lambda c: c and "w3-tag" in c)
            if gloss_span:
                gloss = _norm(gloss_span.get_text(" ", strip=True))
        if ch:
            out.setdefault(key, []).append({"char": ch, "gloss": gloss})
    return out


def _extract_related_words(soup: BeautifulSoup) -> list[dict]:
    """Bottom list of 복합어 tiles. Each has pop_word.asp?word_id=N + word + reading."""
    out = []
    seen_ids: set[str] = set()
    for tile in soup.find_all("div", class_=lambda c: c and "w3-button" in c and "w3-border" in c):
        onclick = tile.get("onclick") or ""
        m = _WORD_ID_RE.search(onclick)
        if not m:
            continue
        wid = m.group(1)
        if wid in seen_ids:
            continue
        # First inner hanja span = the word (may contain highlighted span inside)
        word_span = tile.find("span", class_="hanja1")
        if word_span is None:
            continue
        # Hanzi characters — join without separator so embedded highlight spans
        # don't leave whitespace between glyphs.
        word = "".join(word_span.get_text("", strip=True).split())
        # Reading is the second inline span
        read_span = tile.find("span", class_="hanja11")
        reading = _norm(read_span.get_text(" ", strip=True)) if read_span else None
        if word:
            out.append({"word": word, "reading": reading, "word_id": wid})
            seen_ids.add(wid)
    return out


def parse_one(html: str, cp: int, hex_part: str) -> tuple[dict, list[str]]:
    """Returns (record, anomalies). record always carries cp/hex/char."""
    soup = BeautifulSoup(html, "lxml")
    anomalies: list[str] = []
    rec: dict[str, Any] = {"cp": cp, "hex": hex_part.upper(), "char": chr(cp)}

    # Map label text → extractor function
    # NOTE label text is whitespace-collapsed and may differ slightly per record;
    # we match on the normalized form.
    label_spans = soup.find_all("span", class_=lambda c: c and "w3-light-grey" in c)
    # Build (label, value_div) pairs
    seen_labels: set[str] = set()
    for sp in label_spans:
        label = _label_text(sp)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        val_div = _find_value_div(sp)

        if label == "획 수":
            total, rad_strokes = _extract_stroke_count(val_div)
            if total is not None:
                rec["total_strokes"] = total
            if rad_strokes is not None:
                rec["radical_strokes"] = rad_strokes
        elif label == "부 수":
            rd = _extract_radical(val_div)
            if rd:
                rec["radical"] = rd
        elif label == "분 류":
            cls = _extract_classification(val_div)
            if cls:
                rec["classification"] = cls
        elif label == "한어병음(만다린)":
            p = _extract_pinyin(val_div)
            if p:
                rec["pinyin"] = p
        elif label == "영 문":
            en = _extract_english(val_div)
            if en:
                rec["english"] = en
        elif label == "모양자":
            sh = _extract_shape(val_div)
            if sh:
                rec["shape"] = sh
        elif label == "자 원":
            ety = _extract_etymology(val_div)
            if ety:
                rec["etymology"] = ety
        elif label in ("훈 음", "자 해"):
            # Already in tree.jsonl, silently skip.
            pass
        elif label in _SUPPRESSED_LABELS:
            pass
        else:
            anomalies.append(f"unknown_label:{label}")

    # Non-label blocks
    wu = _extract_word_usage(soup)
    if wu:
        rec["word_usage"] = wu
    rels = _extract_detail_only_relations(soup)
    if rels:
        rec["related_characters"] = rels
    rw = _extract_related_words(soup)
    if rw:
        rec["related_words"] = rw

    # Sanity: detail is considered "content-less" if none of the payload keys
    # ended up in the record. Flag for audit.
    has_any = any(k in rec for k in ("total_strokes", "radical", "classification",
                                       "pinyin", "english", "shape", "etymology",
                                       "word_usage", "related_characters", "related_words"))
    if not has_any:
        anomalies.append("no_content_extracted")

    return rec, anomalies


def _process_file(path_str: str) -> tuple[dict | None, list[str], str | None]:
    """Worker: parse one file. Returns (record, anomalies, hex_404)."""
    p = Path(path_str)
    name = p.name
    if name.endswith(".html"):
        hex_part = name[:-5]
        try:
            cp = int(hex_part, 16)
        except ValueError:
            return None, [f"bad_hex:{hex_part}"], None
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()
        except Exception as e:
            return None, [f"read_error:{e}"], None
        try:
            rec, anoms = parse_one(html, cp, hex_part)
        except Exception as e:
            return None, [f"parse_exception:{type(e).__name__}:{e}"], None
        return rec, anoms, None
    elif name.endswith(".404"):
        return None, [], name[:-4]
    return None, [f"unknown_file:{name}"], None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                     help="only process first N files (smoke test)")
    ap.add_argument("--files", nargs="*", default=None,
                     help="explicit list of files to process (abs paths)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = sorted(IN.iterdir())
    if args.limit:
        files = files[: args.limit]
    n_total = len(files)
    print(f"processing {n_total:,} files with {args.workers} workers")

    ok_count = 0
    miss_count = 0
    anom_count = 0
    t0 = time.perf_counter()

    with open(OUT_JSONL, "w", encoding="utf-8") as fout, \
         open(OUT_404, "w", encoding="utf-8") as fout404, \
         open(OUT_ANOM, "w", encoding="utf-8") as fanom:

        if args.workers <= 1:
            results_iter = (_process_file(str(p)) for p in files)
        else:
            pool = mp.Pool(processes=args.workers)
            results_iter = pool.imap_unordered(_process_file,
                                                (str(p) for p in files),
                                                chunksize=64)

        for i, (rec, anoms, hex_404) in enumerate(results_iter, 1):
            if rec is not None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok_count += 1
            if hex_404 is not None:
                try:
                    cp = int(hex_404, 16)
                    fout404.write(json.dumps({"cp": cp, "hex": hex_404.upper(),
                                               "char": chr(cp)},
                                              ensure_ascii=False) + "\n")
                except ValueError:
                    pass
                miss_count += 1
            if anoms:
                anom_count += 1
                cp_key = rec.get("cp") if rec else None
                fanom.write(json.dumps({"cp": cp_key, "anomalies": anoms},
                                        ensure_ascii=False) + "\n")
            if i % 5000 == 0 or i == n_total:
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (n_total - i) / max(rate, 1e-6)
                print(f"  {i:,}/{n_total:,}  ok={ok_count:,}  404={miss_count}  "
                      f"anom={anom_count}  rate={rate:.0f}/s  eta={eta:.0f}s")

        if args.workers > 1:
            pool.close()
            pool.join()

    print()
    print(f"success={ok_count:,}  404={miss_count}  with_anomalies={anom_count}")
    print(f"wrote: {OUT_JSONL}")
    print(f"wrote: {OUT_404}")
    print(f"wrote: {OUT_ANOM}")


if __name__ == "__main__":
    main()

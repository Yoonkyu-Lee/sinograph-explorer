"""Select the Stage 1 training class list from canonical DB v2.

Philosophy: every char canonical DB knows is a class (proposal's novelty =
language-independent hanzi identity). Tiers differ only in PER-CLASS QUOTA.

Tiers (strict priority cascade — earlier tier wins):
  T1 Mobile-covered (heavy)   : char appears in db_src/e-hanja/hSchool.csv
                                (10,932자, 실생활 Korean OCR 이 실제로 만날
                                char 의 경험적 근사. 주력 학습 대상)
  T2 Common non-mobile        : Unihan Core 2020 ∨ education_level(중/고등용)
                                ∨ hanja_grade(1~8급), T1 이 아님
  T3 Variant fam (corroborated): multi-source edge 포함된 enriched family 의
                                 모든 member, T1~T2 이 아님
  T4 Korean BMP non-mobile    : BMP ∧ ehanja_online ∧ ¬kanjidic2 ∧ ¬mmh,
                                T1~T3 이 아님
  T5 Variant fam (1-source)   : enriched_family_size ≥ 2, T1~T4 이 아님
  T6 Rest (existence only)    : 나머지 모든 char (50 샘플로 embedding 에 자리만)

Quota defaults:
  T1 500 / T2 300 / T3 300 / T4 200 / T5 150 / T6 50

Output:
  sinograph_canonical_v2/out/class_list_v1.jsonl
  stdout: tier + block + family + total-runtime distribution
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILD_ROOT.parent
DB_PATH = BUILD_ROOT / "out" / "sinograph_canonical_v2.sqlite"
OUT_PATH = BUILD_ROOT / "out" / "class_list_v1.jsonl"

# E-hanja mobile app DB — its hSchool table is an empirically-curated set of
# Korean-context everyday hanja. Used as T1 anchor (실생활 OCR 등장 빈도 대리지표).
EHANJA_MOBILE_HSCHOOL_CSV = REPO_ROOT / "db_src" / "e-hanja" / "ejajeon_csv" / "hSchool.csv"


# default quotas (per-class target sample counts). override via CLI.
DEFAULT_QUOTAS = {"T1": 500, "T2": 300, "T3": 300, "T4": 200, "T5": 150, "T6": 50}

# Hanja 검정 grade prefixes that qualify for T2 Common. 특급/특급II 제외.
T2_GRADE_PREFIXES = ("1급", "2급", "3급", "4급", "5급", "6급", "7급", "8급")


def load_mobile_coverage() -> set[str]:
    """Read e-hanja mobile hSchool.csv → set of codepoint strings ('U+XXXX')."""
    import csv
    out: set[str] = set()
    if not EHANJA_MOBILE_HSCHOOL_CSV.exists():
        return out
    with EHANJA_MOBILE_HSCHOOL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ch = (row.get("hanja") or "").strip()
            if len(ch) != 1:
                continue
            out.add(f"U+{ord(ch):04X}")
    return out

# Unicode blocks we care about (for reporting).
BLOCKS = [
    ("CJK_Unified",      0x4E00,  0x9FFF),
    ("CJK_Ext_A",        0x3400,  0x4DBF),
    ("CJK_Compat",       0xF900,  0xFAFF),
    ("Radicals_Supp",    0x2E80,  0x2EFF),
    ("Kangxi_Radicals",  0x2F00,  0x2FDF),
    ("Ext_B_SMP",        0x20000, 0x2A6DF),
    ("Ext_C_SMP",        0x2A700, 0x2B73F),
    ("Ext_D_SMP",        0x2B740, 0x2B81F),
    ("Ext_E_SMP",        0x2B820, 0x2CEAF),
    ("Hiragana",         0x3040,  0x309F),
    ("Katakana",         0x30A0,  0x30FF),
]


def block_of(cp_int: int) -> str:
    for name, lo, hi in BLOCKS:
        if lo <= cp_int <= hi:
            return name
    return "Other"


def build_class_list(quotas: dict[str, int]) -> tuple[list[dict], dict]:
    if not DB_PATH.exists():
        raise SystemExit(f"canonical v2 SQLite not found: {DB_PATH}")

    db = sqlite3.connect(DB_PATH)

    # --- load full records ---
    print(f"[select] loading canonical records…")
    records: dict[str, dict] = {}
    for (data_json,) in db.execute("SELECT data_json FROM characters"):
        rec = json.loads(data_json)
        records[rec["codepoint"]] = rec
    print(f"[select]   {len(records):,} chars")

    # --- load multi-source edges (for T2 'corroborated' gate) ---
    multi_source_cps: set[str] = set()
    for s, t in db.execute(
        "SELECT source_codepoint, target_codepoint FROM variant_edges WHERE support_count >= 2"
    ):
        multi_source_cps.add(s)
        multi_source_cps.add(t)
    print(f"[select]   multi-source edge endpoints: {len(multi_source_cps):,} chars")

    # --- Tier 1: Mobile-covered ---
    mobile_cps = load_mobile_coverage()
    print(f"[select]   e-hanja mobile coverage: {len(mobile_cps):,} chars")
    t1 = {cp for cp in mobile_cps if cp in records}

    # --- Tier 2: Common non-mobile ---
    t2: set[str] = set()
    for cp, rec in records.items():
        excl_u = (rec.get("source_exclusive") or {}).get("unihan") or {}
        excl_e = (rec.get("source_exclusive") or {}).get("ehanja_online") or {}
        cls = excl_e.get("classification") or {}
        if excl_u.get("unihan_core_2020"):
            t2.add(cp); continue
        edu = cls.get("education_level", "")
        if edu in ("중학용", "고등용"):
            t2.add(cp); continue
        grade = (cls.get("hanja_grade") or "").strip()
        if grade.startswith(T2_GRADE_PREFIXES):
            t2.add(cp)

    # --- Tier 3: Variant family (corroborated) ---
    # A family (one enriched connected component) qualifies if ANY member of
    # it appears in a multi-source (support_count≥2) edge.
    families_by_rep: dict[str, list[str]] = {}
    for cp, rec in records.items():
        vg = rec.get("variant_graph") or {}
        fm = vg.get("enriched_family_members") or []
        if len(fm) < 2:
            continue
        rep = vg.get("enriched_representative_form") or sorted(fm)[0]
        families_by_rep.setdefault(rep, fm)
    corroborated_reps: set[str] = set()
    for rep, members in families_by_rep.items():
        if any(m in multi_source_cps for m in members):
            corroborated_reps.add(rep)
    t3: set[str] = set()
    for rep in corroborated_reps:
        t3.update(families_by_rep[rep])

    # --- Tier 4: Korean region BMP (non-mobile, non-common) ---
    t4: set[str] = set()
    for cp, rec in records.items():
        cp_int = int(cp[2:], 16)
        if cp_int >= 0x10000:
            continue
        sf = rec.get("source_flags") or {}
        if not sf.get("ehanja_online"):
            continue
        if sf.get("kanjidic2") or sf.get("makemeahanzi"):
            continue
        t4.add(cp)

    # --- Tier 5: Variant family (1-source) ---
    t5: set[str] = set()
    for cp, rec in records.items():
        vg = rec.get("variant_graph") or {}
        if len(vg.get("enriched_family_members") or []) < 2:
            continue
        t5.add(cp)

    # --- Tier 6: Rest ---
    t6: set[str] = set(records.keys())

    # --- Build final class list. Earlier tier wins. ---
    chosen: list[dict] = []
    tier_counts: Counter = Counter()
    for cp in sorted(records.keys()):
        tiers_hit = []
        if cp in t1: tiers_hit.append("T1")
        if cp in t2: tiers_hit.append("T2")
        if cp in t3: tiers_hit.append("T3")
        if cp in t4: tiers_hit.append("T4")
        if cp in t5: tiers_hit.append("T5")
        tiers_hit.append("T6")  # everyone in T6

        picked_tier = tiers_hit[0]
        samples = quotas[picked_tier]
        # quota=0 means "drop this tier from the class list entirely"
        if samples <= 0:
            continue
        rec = records[cp]
        vg = rec.get("variant_graph") or {}
        chosen.append({
            "codepoint": cp,
            "char": rec["character"],
            "tiers": tiers_hit,
            "tier_picked": picked_tier,
            "target_samples": samples,
            "enriched_family_size": len(vg.get("enriched_family_members") or [cp]),
            "enriched_representative": vg.get("enriched_representative_form"),
            "block": block_of(int(cp[2:], 16)),
            "source_flags": rec.get("source_flags"),
        })
        tier_counts[picked_tier] += 1

    stats = {
        "t1_size": len(t1), "t2_size": len(t2), "t3_size": len(t3),
        "t4_size": len(t4), "t5_size": len(t5), "t6_size": len(t6),
        "union_size": len(records),
        "picked_tier_counts": dict(tier_counts),
        "corroborated_families": len(corroborated_reps),
        "multi_source_edge_endpoints": len(multi_source_cps),
        "mobile_coverage": len(mobile_cps),
        "quotas": quotas,
    }
    return chosen, stats


def report(chosen: list[dict], stats: dict, rate_per_s: float) -> None:
    print()
    print(f"=== raw tier sizes (overlapping before priority cascade) ===")
    print(f"  T1 Mobile-covered        : {stats['t1_size']:>7,}   "
          f"(mobile csv: {stats['mobile_coverage']:,})")
    print(f"  T2 Common non-mobile     : {stats['t2_size']:>7,}")
    print(f"  T3 Variant fam (corrob.) : {stats['t3_size']:>7,}")
    print(f"  T4 Korean BMP non-mobile : {stats['t4_size']:>7,}")
    print(f"  T5 Variant fam (1-src)   : {stats['t5_size']:>7,}")
    print(f"  T6 Rest (everyone)       : {stats['t6_size']:>7,}")

    print()
    print(f"=== picked-tier distribution (earlier tier wins, mutually exclusive) ===")
    for tier in ("T1", "T2", "T3", "T4", "T5", "T6"):
        n = stats["picked_tier_counts"].get(tier, 0)
        quota = stats["quotas"][tier]
        subtotal = n * quota
        print(f"  {tier:5s} n={n:>7,}   quota={quota:>4}/class   subtotal={subtotal:>12,}")

    total = sum(c["target_samples"] for c in chosen)
    print()
    print(f"TOTAL unique classes : {len(chosen):,}")
    print(f"TOTAL samples        : {total:,}")
    runtime_s = total / rate_per_s
    print(f"Est. runtime @ {rate_per_s:.0f} s/s steady: "
          f"{runtime_s/3600:.1f} h  ({runtime_s:.0f} s)")
    bytes_est = total * 45 * 1024
    print(f"Est. disk @ 45 KB/PNG: {bytes_est/1e9:.1f} GB")

    print()
    print(f"=== block distribution ===")
    block_count: Counter = Counter()
    block_samples: Counter = Counter()
    for c in chosen:
        block_count[c["block"]] += 1
        block_samples[c["block"]] += c["target_samples"]
    for name, _, _ in BLOCKS + [("Other", 0, 0)]:
        if block_count.get(name, 0) == 0:
            continue
        print(f"  {name:<18s} classes={block_count[name]:>6,}  samples={block_samples[name]:>9,}")

    print()
    print(f"=== family size distribution (T2 chars only) ===")
    t2_chars = [c for c in chosen if "T2" in c["tiers"]]
    fam_sizes = Counter(c["enriched_family_size"] for c in t2_chars)
    buckets = [(2,3), (3,5), (5,10), (10,20), (20,60)]
    for lo, hi in buckets:
        n = sum(v for s, v in fam_sizes.items() if lo <= s < hi)
        print(f"  T2 chars in families [{lo:>2}, {hi:>2}): {n:>6,}")

    print()
    print(f"saved class list to: {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", type=int, default=DEFAULT_QUOTAS["T1"])
    ap.add_argument("--t2", type=int, default=DEFAULT_QUOTAS["T2"])
    ap.add_argument("--t3", type=int, default=DEFAULT_QUOTAS["T3"])
    ap.add_argument("--t4", type=int, default=DEFAULT_QUOTAS["T4"])
    ap.add_argument("--t5", type=int, default=DEFAULT_QUOTAS["T5"])
    ap.add_argument("--t6", type=int, default=DEFAULT_QUOTAS["T6"])
    ap.add_argument("--rate", type=float, default=178.0,
                     help="assumed v3 engine throughput for runtime estimate")
    args = ap.parse_args()
    quotas = {"T1": args.t1, "T2": args.t2, "T3": args.t3,
              "T4": args.t4, "T5": args.t5, "T6": args.t6}

    chosen, stats = build_class_list(quotas)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    report(chosen, stats, args.rate)


if __name__ == "__main__":
    main()

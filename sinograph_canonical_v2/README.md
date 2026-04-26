# Sinograph Canonical DB v2

v1 (`sinograph_canonical_v1/`) 의 후속 버전. e-hanja 모바일 DB (10,932자) 를
**e-hanja_online (76,013자)** 로 교체하고, 레코드 구조를 **교집합 / 합집합−교집합**
원칙에 맞춰 재정렬했다.

자세한 설계는 [`doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md`](../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md).
레코드 스키마는 [`schema/canonical_schema_v2.md`](schema/canonical_schema_v2.md).
빌드 산출물 사용법은 [`CANONICAL_DB_V2_MANUAL.md`](CANONICAL_DB_V2_MANUAL.md).

## 한 줄 변경 요약

v1: Unihan backbone × e-hanja mobile 10,932 + KANJIDIC2 + MMH (w/ stroke media)
v2: Unihan backbone × **e-hanja_online 76,013** + KANJIDIC2 + MMH (structure only, no stroke media)

## 디렉토리 구조

```
sinograph_canonical_v2/
  README.md                         ← 이 파일
  CANONICAL_DB_V2_MANUAL.md         ← 산출물 인벤토리 + 사용 가이드
  schema/
    canonical_schema_v2.md          ← 레코드 shape + authority 정책
  scripts/
    build_canonical_db_v2.py        ← ETL 전부 (adapters + merge + projection + SQLite)
    lookup_canonical_db_v2.py       ← CLI lookup 유틸
    analyze_canonical_db_v2.py      ← 커버리지 / corroboration 집계
    compare_v1_v2.py                ← v1 ↔ v2 diff report
  staging/                          ← source adapter 정규화 output
    unihan.normalized.jsonl
    ehanja_online.normalized.jsonl
    kanjidic2.normalized.jsonl
    makemeahanzi.normalized.jsonl
  out/                              ← canonical 최종 산출물
    canonical_characters.jsonl
    canonical_variants.jsonl
    variant_components.jsonl
    sinograph_canonical_v2.sqlite
    build_summary.json
  tests/                            ← 샘플/parity 검증 스크립트
```

## 빌드

```bash
python scripts/build_canonical_db_v2.py
```

입력은 모두 `../db_src/` 에서. 출력은 위 `out/` + `staging/`.

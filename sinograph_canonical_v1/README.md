# Sinograph Canonical DB v1 Workspace

> ⚠️ **DEPRECATED — superseded by v2.**
> 이 디렉토리는 `sinograph_canonical_v2/` 가 출시되면서 **참고/아카이브** 용으로만
> 유지된다. **신규 소비자는 `sinograph_canonical_v2/` 를 사용해야 한다.**
>
> v1 대비 v2 의 핵심 변경:
> - `e-hanja` (mobile DB, 10,932자) → **`e-hanja_online` (76,013자, +6.9×)**
> - variant edges 28,118 → **88,740 (+3.16×)** + **multi-source corroboration signal** (2,805 edges)
> - `media.stroke_svg_paths` / `media.stroke_medians` 제거 (OCR 합성은 v3 엔진이 `db_src/` 에서 직접 접근)
> - record shape 재설계: `core` + `provenance` + `core_alternatives` + `source_exclusive` 3-layer
> - Family graph 에서 의미 외곽 relation (유의/상대/별자) 제외 — max cluster 폭발 방지
>
> 자세한 이관 사유·설계: [`../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md`](../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md)
> v2 workspace: [`../sinograph_canonical_v2/`](../sinograph_canonical_v2/)
>
> 이 폴더의 `out/sinograph_canonical_v1.sqlite` 등 산출물은 그대로 보존된다
> (다운스트림 마이그레이션 완료 후 archive 이동 검토).

---

이 디렉토리는 **통합 Sinograph DB v1 구현 전용 작업 공간**이다.

핵심 원칙:

- `lab3/db_src/` 아래의 원본 DB 소스는 **절대 수정하지 않는다**
- canonical merge / normalization / graph building / SQLite export는 모두 이 디렉토리 안에서만 수행한다
- 생성물도 가능하면 이 디렉토리 내부에만 둔다

즉 역할 분리는 다음과 같다.

- `lab3/db_src/`
  - 원천 source-of-truth
  - 읽기 전용처럼 취급
- `lab3/sinograph_canonical_v1/`
  - 통합 스키마
  - ETL 스크립트
  - staging JSONL
  - canonical JSONL
  - SQLite 산출물

## Recommended Layout

```text
sinograph_canonical_v1/
  README.md
  scripts/
  schema/
  staging/
  out/
  tests/
```

## Planned Usage

- source adapters:
  - `scripts/`
- canonical schema / mapping notes:
  - `schema/`
- temporary normalized per-source JSONL:
  - `staging/`
- final canonical JSONL + SQLite:
  - `out/`
- spot-check / regression checks:
  - `tests/`

이 워크스페이스 안의 코드는 `../db_src/...`를 읽고, 결과만 이 디렉토리 아래에 쓴다.

## Variant Policy

v1.1부터 variant family는 두 개의 parallel view를 가진다.

- **canonical family**
  - Unihan-only authoritative backbone
  - 기존 `variants.family_members`, `variants.representative_form`
- **enriched family**
  - Unihan + supplementary e-hanja relations + resolvable KANJIDIC2 variant refs
  - `variant_graph.enriched_family_members`, `variant_graph.enriched_representative_form`

즉 기존 consumer는 계속 canonical view만 읽어도 되고,
새 consumer는 enriched family를 추가로 사용할 수 있다.

## Build

권장 빌드 명령:

```powershell
python .\lab3\sinograph_canonical_v1\scripts\build_canonical_db.py
```

SQLite 없이 JSONL만 만들고 싶다면:

```powershell
python .\lab3\sinograph_canonical_v1\scripts\build_canonical_db.py --skip-sqlite
```

산출물 위치:

- source-normalized staging JSONL:
  - `sinograph_canonical_v1/staging/`
- canonical JSONL / summary / SQLite:
  - `sinograph_canonical_v1/out/`

주요 산출물:

- `out/canonical_characters.jsonl`
  - 문자 1자당 1 row canonical record
- `out/canonical_variants.jsonl`
  - canonical + supplementary typed edge list
- `out/variant_components.jsonl`
  - canonical / enriched family summary
- `out/sinograph_canonical_v1.sqlite`
  - app/query용 SQLite projection

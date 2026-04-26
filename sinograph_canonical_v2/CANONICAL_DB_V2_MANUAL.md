# Sinograph Canonical DB v2 Manual

이 문서는 `sinograph_canonical_v2/` 에서 빌드되는 통합 Sinograph canonical DB v2
의 **실제 산출물 구조와 읽는 방법** 을 설명한다.

- 설계 배경·이관 사유: [`../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md`](../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md)
- 레코드 스키마: [`schema/canonical_schema_v2.md`](schema/canonical_schema_v2.md)
- 빌드 스크립트: [`scripts/build_canonical_db_v2.py`](scripts/build_canonical_db_v2.py)

---

## 1. 한 줄 요약

**v2 = 자전 (lookup + variant graph + metadata)** 전용 backbone.
- `e-hanja_online` (76,013자) 중심
- OCR stroke geometry 는 v2 **밖** — `db_src/` 에서 v3 엔진이 직접 소비
- 3-layer data model: `core` (교집합) + `provenance`/`core_alternatives` (공급 출처 audit) + `source_exclusive.<source>.*` (한 소스 고유)

## 2. Build Snapshot (2026-04-19)

| 지표 | 값 |
|---|---:|
| canonical character rows | **103,327** |
| source record counts | unihan 102,998 / **ehanja_online 76,013** / kanjidic2 13,108 / makemeahanzi 9,574 |
| variant edges (total) | **88,740** |
| &nbsp;&nbsp;canonical (Unihan) | 17,902 |
| &nbsp;&nbsp;supplementary | 70,838 |
| multi-source corroborated edges | **2,805** |
| canonical variant components | 94,656 |
| enriched variant components | 70,712 |
| max canonical component size | 11 |
| **max enriched component size** | **60** |
| SMP characters with variant evidence | 22,050 |

v1 대비 증분: ehanja **6.95×**, variant edges **3.16×**, multi-source corroboration = 신규 signal.

## 3. Directory / Artifact Inventory

```
sinograph_canonical_v2/
  README.md
  CANONICAL_DB_V2_MANUAL.md          ← 이 파일
  schema/
    canonical_schema_v2.md           ← shape + authority 정책
  scripts/
    build_canonical_db_v2.py         ← ETL pipeline 전체
  staging/                           ← source adapter 정규화 출력
    unihan.normalized.jsonl          (102,998)
    ehanja_online.normalized.jsonl   (76,013)
    kanjidic2.normalized.jsonl       (13,108)
    makemeahanzi.normalized.jsonl    (9,574)
  out/                               ← 최종 canonical 산출물
    canonical_characters.jsonl       (103,327 rows)
    canonical_variants.jsonl         (88,740 edge rows)
    variant_components.jsonl         (103,327 rows, 1-per-char view)
    sinograph_canonical_v2.sqlite    (query/app 사용)
    build_summary.json
  tests/                             ← 검증 스크립트 (선택)
```

## 4. `canonical_characters.jsonl`

1 line = 1 char record (primary key = `codepoint`). Top-level shape:

```
character
codepoint                          "U+XXXX"
source_flags                       {unihan, ehanja_online, kanjidic2, makemeahanzi: bool}

core                               ← 교집합 (≥2 소스 공급 가능)
  radical                          str | None
  total_strokes                    int | None
  definitions.english              [str]
  readings.{mandarin, cantonese, korean_hangul, japanese_on, japanese_kun, vietnamese}  [str]

provenance                         ← core 각 필드의 실제 pick source
  e.g. "readings.korean_hangul": "ehanja_online"

core_alternatives                  ← picked 이외 소스가 같은 필드에 준 값 (corroboration / conflict)
  e.g. "total_strokes": {"kanjidic2": 22, "ehanja_online": 22}

variants                           ← Unihan canonical backbone
  traditional / simplified / semantic / specialized_semantic / z_variants / spoofing
  representative_form
  family_members

supplementary_variants             ← source-explicit relation buckets (codepoint arrays)
  ehanja_dongja / bonja / sokja / yakja / goja / waja / simple / kanji / tongja / hDup
  ehanja_synonyms / opposites / alt_forms    (graph 에서 제외, record 보존)
  kanjidic2_resolved

variant_graph                      ← canonical + enriched 병렬 family view
  canonical_family_members
  canonical_representative_form
  enriched_family_members           ← 제외 relation (synonyms/opposites/alt_forms) 적용 후
  enriched_representative_form

source_exclusive                   ← 한 소스 고유 정보, 경로가 곧 provenance
  unihan
    rs_unicode / kangxi / irg_kangxi / hanyu / unihan_core_2020
  ehanja_online
    korean_hun / korean_explanation / classification / shape / etymology /
    word_usage / related_words / radical_detail / svg_type / stroke_count_animated
  kanjidic2
    korean_romanized / dictionary_refs / query_codes / codepoint_refs / nanori /
    unresolved_variant_refs
  makemeahanzi
    decomposition / etymology_type / etymology_hint / phonetic_component /
    semantic_component

source_payloads                    ← 변환 중 손실 방지용 near-raw
  unihan_raw / ehanja_online_raw / kanjidic2_raw / makemeahanzi_raw
```

### 해석 요령
- `core.*` 필드의 **어느 소스가 줬는지** → `provenance.*` 참고
- **다른 소스가 같은 의미로 준 값** (corroboration 또는 conflict) → `core_alternatives.*`
- **한 소스 고유 정보** 는 항상 `source_exclusive.<source>.<field>` 에만 있음
- variant 관계는 `variants.*` (Unihan 만), `supplementary_variants.*` (그 외), `variant_graph.*` (합쳐진 뷰)

### 예시 — 鑑 (U+9451)

```json
{
  "core": {
    "radical": "167",
    "total_strokes": 22,
    "definitions": {"english": ["mirror, looking glass; reflect"]},
    "readings": {
      "mandarin": ["jiàn"], "cantonese": ["gaam3"],
      "korean_hangul": ["감"],
      "japanese_on": ["カン"], "japanese_kun": ["かんが.みる", "かがみ"],
      "vietnamese": ["Giám"]
    }
  },
  "provenance": {
    "readings.korean_hangul": "ehanja_online",
    "readings.japanese_on": "kanjidic2",
    "readings.mandarin": "unihan",
    ...
  },
  "core_alternatives": {
    "readings.mandarin": {"kanjidic2": ["ジュン"], "makemeahanzi": ["jiàn"]},
    "total_strokes": {"kanjidic2": 22, "ehanja_online": 22}
  },
  "variants": {
    "traditional": [], "simplified": [], "semantic": ["U+9452"],
    "family_members": ["U+9452", "U+9274", "U+9451", ...]
  },
  "supplementary_variants": {
    "ehanja_dongja": ["U+9373", "U+9452", "U+946C", "U+28C32"],
    "ehanja_synonyms": ["U+93E1"],
    "kanjidic2_resolved": ["U+9452", "U+9373"]
  },
  "variant_graph": {
    "canonical_family_members": [...5 members...],
    "enriched_family_members":  [...12 members...]
  },
  "source_exclusive": {
    "ehanja_online": {
      "korean_hun": ["거울"],
      "korean_explanation": [{"meaning": "거울", "root_snd": "감", ...}, ...],
      "classification": {"education_level": "고등용", "hanja_grade": "3급II(2급)", "name_use": "인명용"},
      "shape": {"representative": {"char": "鑑", "gloss": "거울 감"}, "components": [...]},
      "etymology": {"type": "형성문자", "description": "..."},
      "word_usage": "...",
      "related_words": [{"word": "鑑賞", "reading": "감상", "word_id": "3149"}, ...],
      ...
    },
    "makemeahanzi": {"decomposition": "⿰金監", ...},
    "unihan": {"rs_unicode": ["167.14"], ...},
    "kanjidic2": {"nanori": ["..."], ...}
  },
  "source_payloads": {...}
}
```

## 5. `canonical_variants.jsonl` (+ SQLite `variant_edges`)

1 line = 1 variant edge. `(source_codepoint, target_codepoint, relation_scope)` 는 고유 — 같은 pair 를 여러 소스가 지지하면 merge.

```
source_character         str
source_codepoint         "U+XXXX"
target_character         str
target_codepoint         "U+XXXX"
relation_scope           "canonical" | "supplementary"
relation                 primary (first-seen) relation label
sources                  [source_name, ...]             ← support 하는 소스 목록
source_relations         {source_name: original_relation_label}
support_count            int                            ← len(sources). 신뢰도 tier.
```

**모든** relation 의 edge 는 여기 저장됨 (synonyms/opposites/alt_forms 포함).
단 `variant_graph.*_family_members` 계산 시에는 아래 제외 규칙이 적용됨.

### Family Graph Exclusion 규칙
```
FAMILY_GRAPH_EXCLUDED_RELATIONS = {
    "ehanja_synonyms",   # 유의자
    "ehanja_opposites",  # 상대자
    "ehanja_alt_forms",  # 별자 (detail-only)
}
```
쿼리 시 `WHERE relation NOT IN (...)` 으로 가능.

## 6. `variant_components.jsonl`

1 line = 1 codepoint 의 component view.

```
codepoint / character
representative_form              (canonical backward-compat alias)
component_size                   (canonical family size)
family_members                   (canonical, backward-compat)
canonical_representative_form
canonical_family_members
enriched_representative_form
enriched_family_members          (synonyms/opposites/alt_forms 제외 후 계산)
```

## 7. `sinograph_canonical_v2.sqlite`

다운스트림 app 용. Table 목록:

| 테이블 | 역할 |
|---|---|
| `characters` | 1 row / char + `data_json` 에 full record |
| `character_readings` | reading 1 row / value (cp, reading_type, value) |
| `character_meanings` | meaning 1 row / value (cp, language, value). 한국어 설명 포함 |
| `variant_edges` | 1 row / edge. sources_json, source_relations_json, support_count 포함 |
| `variant_components` | 1 row / cp. canonical / enriched family view |
| `source_presence` | 1 row / cp. 4 source flag (0/1) |
| `core_provenance` | `(cp, field_path, source_name)` — 어느 core 값이 어디서 왔는지 |
| `core_alternatives` | `(cp, field_path, source_name, value_json)` — picked 이외 소스의 값 |
| `source_exclusive` | `(cp, source_name, field_path, value_json)` — 한 소스 고유 데이터 |
| `source_payloads` | `(cp, source_name, payload_json)` — near-raw audit |

### 쿼리 예시

**1. 한 글자의 기본 정보**
```sql
SELECT data_json FROM characters WHERE codepoint = 'U+9451';
```

**2. 이 글자의 이체자 (high-confidence = 2+ 소스 동의)**
```sql
SELECT target_character, target_codepoint, relation, support_count, sources_json
FROM variant_edges
WHERE source_codepoint = 'U+9451'
  AND relation NOT IN ('ehanja_synonyms', 'ehanja_opposites', 'ehanja_alt_forms')
ORDER BY support_count DESC, relation;
```

**3. 이 글자의 family (enriched, variant 만)**
```sql
SELECT enriched_family_members_json FROM variant_components WHERE codepoint = 'U+9451';
```

**4. 특정 필드의 source 알아보기**
```sql
SELECT field_path, source_name FROM core_provenance WHERE codepoint = 'U+9451';
```

**5. 값 충돌 감사 (다른 소스가 다른 값을 준 경우)**
```sql
SELECT cp.codepoint, cp.character, ca.field_path, ca.source_name, ca.value_json
FROM core_alternatives ca JOIN characters cp USING (codepoint)
WHERE ca.field_path = 'total_strokes'
  AND ca.value_json != cast(json_extract(cp.data_json, '$.core.total_strokes') as text);
```

**6. SMP 희귀 한자 중 Korean 관계 있는 char**
```sql
SELECT DISTINCT ve.source_codepoint, ve.source_character
FROM variant_edges ve
WHERE ve.relation LIKE 'ehanja_%'
  AND cast(substr(ve.source_codepoint, 3) as integer) >= 0x20000;
```

## 8. 빌드

```bash
python scripts/build_canonical_db_v2.py                   # full build (≈ 실행 90초)
python scripts/build_canonical_db_v2.py --skip-staging    # staging 재사용
```

입력:
- `../db_src/Unihan/Unihan_txt/*.txt`
- `../db_src/e-hanja_online/tree.jsonl` + `detail.jsonl` + `strokes_manifest.jsonl`
- `../db_src/KANJIDIC2/KANJIDIC2_xml/kanjidic2.xml`
- `../db_src/MAKEMEAHANZI/dictionary.txt`

출력은 섹션 3 참고.

## 9. v1 과의 호환성 노트

- `characters` 테이블의 **컬럼 집합이 바뀜** (`character_media` 테이블 삭제,
  `core_provenance` / `core_alternatives` / `source_exclusive` 추가). 기존
  consumer 는 JSON 경로 또는 테이블 이름만 교체하면 됨.
- `source_flags.ehanja` → **`source_flags.ehanja_online`** (이름 교체)
- `variants.*` 의 의미는 유지 (Unihan canonical backbone)
- `variant_graph.enriched_family_members` 는 **FAMILY_GRAPH_EXCLUDED_RELATIONS**
  제외 후 계산 — v1.1 대비 더 엄격한 정의이지만 cluster 폭발 방지
- `media.stroke_svg_paths` / `media.stroke_medians` **완전 제거** — OCR 합성
  엔진 (v3) 가 `db_src/` 에서 직접 접근

## 10. 알려진 한계

- `core` 에 잡혀있는 필드도 e-hanja_online 은 char 에 따라 구체성이 다름.
  예: `ehanja_online.total_strokes` 는 animated SVG 가 있는 char 에만 존재. static
  SVG (Ext B SMP 등) 의 경우 detail 페이지에서 파싱 (여전히 대부분 수집됨).
- `multi_source_edge_count = 2,805` 는 downstream 에서 "2+ 소스 corroborated"
  filter 로 쓸 수 있지만 절대 수는 적음. 이는 소스마다 variant 분류 체계가
  다르기 때문 (e.g. Unihan `kSemanticVariant` 와 e-hanja `dongja` 가 같은 pair
  를 지지하는 경우가 전체의 3% 정도).
- `representative_form` 은 project heuristic 이지 philological 정답 아님.

## 11. 변경 이력

- 2026-04-19 — 초안. v1.1 + e-hanja_online 완전 통합 + 3-layer data model +
  FAMILY_GRAPH_EXCLUDED_RELATIONS 정책 완성.

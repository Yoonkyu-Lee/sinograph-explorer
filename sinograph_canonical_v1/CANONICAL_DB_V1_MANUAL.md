# Sinograph Canonical DB v1 Manual

> ⚠️ **DEPRECATED — superseded by v2.** 이 문서는 v1 스키마의 역사적 레퍼런스로만
> 유지된다. 신규 소비자는 `../sinograph_canonical_v2/CANONICAL_DB_V2_MANUAL.md`
> 를 참조할 것. 이관 사유·설계는 [`../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md`](../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md).

이 문서는 `lab3/sinograph_canonical_v1/`에서 생성되는 **통합 Sinograph canonical DB v1**의
실제 산출물 구조를 설명한다.

이 DB는 다음 4개 source를 합쳐 만든다.

- `Unihan`
- `e-hanja`
- `KANJIDIC2`
- `MakeMeAHanzi`

핵심 철학은 다음과 같다.

- **intersection + extensions**
  - 공통적으로 맞출 수 있는 정보는 canonical core에 둔다.
  - source별 고유 richness는 `references`와 `source_payloads`에 남긴다.
- **JSONL first, SQLite derived**
  - JSONL이 canonical intermediate이자 inspection-friendly artifact다.
  - SQLite는 앱 / query / graph viewer 용 파생 산출물이다.
- **source DB는 수정하지 않는다**
  - 원본 source는 `lab3/db_src/` 아래에 남기고, canonical build는 별도 워크스페이스에서만 수행한다.

---

## 1. Build Workspace Layout

```text
sinograph_canonical_v1/
  README.md
  CANONICAL_DB_V1_MANUAL.md
  schema/
    canonical_schema_v1.md
  scripts/
    build_canonical_db.py
  staging/
    unihan.normalized.jsonl
    ehanja.normalized.jsonl
    kanjidic2.normalized.jsonl
    makemeahanzi.normalized.jsonl
  out/
    canonical_characters.jsonl
    canonical_variants.jsonl
    variant_components.jsonl
    build_summary.json
    sinograph_canonical_v1.sqlite
```

---

## 2. Current Build Snapshot

현재 빌드 결과(`build_summary.json`) 기준:

- canonical characters: `103,021`
- canonical variant components: `94,350`
- enriched variant components: `91,270`
- variant edges: `28,118`
- canonical edges: `18,586`
- supplementary edges: `9,532`
- largest canonical component size: `11`
- largest enriched component size: `23`
- characters with enriched family growth: `7,385`

source record counts:

- `Unihan`: `102,998`
- `e-hanja`: `10,932`
- `KANJIDIC2`: `13,108`
- `MakeMeAHanzi`: `9,574`

즉 이 canonical DB는 **거의 전체 Unicode Han backbone은 Unihan이 들고 가고**,
다른 3개 source가 읽기 / 한국어 설명 / decomposition / stroke graphics를 채워넣는 구조다.

---

## 3. Artifact Inventory

### 3.1 Staging JSONL

staging layer는 source별 normalized per-character row를 담는다.

- [unihan.normalized.jsonl](./staging/unihan.normalized.jsonl)
- [ehanja.normalized.jsonl](./staging/ehanja.normalized.jsonl)
- [kanjidic2.normalized.jsonl](./staging/kanjidic2.normalized.jsonl)
- [makemeahanzi.normalized.jsonl](./staging/makemeahanzi.normalized.jsonl)

공통 목적:

- 원본 형식을 source-neutral한 per-character JSON으로 맞춘다.
- merge 전에 source adapter 결과를 inspection / diff할 수 있게 한다.

공통 record shape:

```text
NormalizedSourceRecord
  source_name
  character
  codepoint
  radical
  total_strokes
  readings
  definitions
  variants
  supplementary_variants
  structure
  media
  references
  source_payload
```

이 단계에서는 source별 특성이 아직 많이 남아 있다.

### 3.2 Canonical JSONL

최종 canonical outputs:

- [canonical_characters.jsonl](./out/canonical_characters.jsonl)
- [canonical_variants.jsonl](./out/canonical_variants.jsonl)
- [variant_components.jsonl](./out/variant_components.jsonl)
- [build_summary.json](./out/build_summary.json)

### 3.3 SQLite

- [sinograph_canonical_v1.sqlite](./out/sinograph_canonical_v1.sqlite)

이 파일은 viewer / lookup / downstream query에서 직접 사용하도록 설계됐다.

---

## 4. `canonical_characters.jsonl`

이 파일은 **문자 1자당 1 row**를 가진 canonical record 집합이다.

각 줄은 JSON object 하나이며, practical primary key는 `codepoint`다.

### 4.1 Top-level fields

각 row의 top-level fields:

- `character`
  - 실제 한 글자 literal
- `codepoint`
  - `U+XXXX` 형식
  - canonical join key
- `source_flags`
  - source별 presence flag
- `core`
  - 공통적으로 정렬한 핵심 정보
- `variants`
  - variant family 정보
- `structure`
  - decomposition / etymology
- `media`
  - stroke path / medians
- `references`
  - source별 reference block
- `source_payloads`
  - 각 source의 raw / near-raw payload

### 4.2 `source_flags`

구조:

```text
source_flags
  unihan: bool
  ehanja: bool
  kanjidic2: bool
  makemeahanzi: bool
```

역할:

- 현재 row가 어느 source에서 관찰되었는지 표시
- 교집합 / coverage 분석의 기본 지표

예:

```json
"source_flags": {
  "ehanja": true,
  "kanjidic2": true,
  "makemeahanzi": true,
  "unihan": true
}
```

### 4.3 `core`

canonical core는 다음 구조를 가진다.

```text
core
  radical
  total_strokes
  definitions
    english
    korean_explanation
    korean_hun
  readings
    mandarin
    cantonese
    korean_hangul
    korean_romanized
    japanese_on
    japanese_kun
    vietnamese
```

모든 reading / definition field는 **항상 array**다.

#### `core.radical`
- 가능한 경우 radical id / radical number
- authority order:
  - `Unihan`
  - `KANJIDIC2 classical radical`
  - `e-hanja busu`

#### `core.total_strokes`
- 정수 stroke count
- authority order:
  - `Unihan`
  - `KANJIDIC2`
  - `e-hanja`
  - `MakeMeAHanzi`

#### `core.definitions.english`
- canonical English definition list
- fill order:
  - `Unihan.kDefinition`
  - `KANJIDIC2 meaning`
  - `MakeMeAHanzi.definition`

#### `core.definitions.korean_explanation`
- canonical Korean explanation
- source:
  - `e-hanja.hRoot.rMeaning`

#### `core.definitions.korean_hun`
- canonical Korean hun / semantic gloss
- source:
  - `e-hanja.hRead`

#### `core.readings.*`

fill policy:

- `mandarin`
  - `Unihan.kMandarin`
  - fallback `KANJIDIC2 pinyin`
  - fallback `MakeMeAHanzi pinyin`
- `cantonese`
  - `Unihan.kCantonese`
- `korean_hangul`
  - `e-hanja hSnd / hRoot.rSnd`
  - fallback `KANJIDIC2 korean_h`
- `korean_romanized`
  - `KANJIDIC2 korean_r`
- `japanese_on`
  - `KANJIDIC2 ja_on`
  - fallback `Unihan.kJapaneseOn`
- `japanese_kun`
  - `KANJIDIC2 ja_kun`
  - fallback `Unihan.kJapaneseKun`
- `vietnamese`
  - `Unihan.kVietnamese`
  - fallback `KANJIDIC2 vietnam`

### 4.4 `variants`

구조:

```text
variants
  traditional
  simplified
  semantic
  specialized_semantic
  z_variants
  spoofing
  representative_form
  family_members
```

`traditional`, `simplified`, `semantic`, `specialized_semantic`, `z_variants`, `spoofing`은 모두
**codepoint array**다.

authority:
- variant graph는 `Unihan`만을 backbone으로 사용한다.

#### `representative_form`
- 해당 variant component의 대표형
- philological truth가 아니라 project heuristic
- type: `U+XXXX`

#### `family_members`
- 현재 문자가 속한 undirected variant component의 전체 member codepoint list
- 항상 자기 자신은 포함한다

즉 `variants.*`는 v1.1 이후에도 **canonical / Unihan-authoritative view**다.

### 4.5 `supplementary_variants`

구조:

```text
supplementary_variants
  ehanja_yakja
  ehanja_bonja
  ehanja_simple_china
  ehanja_kanji
  ehanja_dongja
  ehanja_tongja
  kanjidic2_resolved
```

모든 값은 codepoint array다.

역할:

- canonical relation semantics를 바꾸지 않고
- e-hanja / KANJIDIC2가 제공하는 supplementary variant evidence를 보존

주의:

- 이 블록은 Unihan `traditional/simplified/semantic` bucket으로 재분류되지 않는다.
- source field 이름을 그대로 드러내는 것이 정책이다.

### 4.6 `variant_graph`

구조:

```text
variant_graph
  canonical_family_members
  canonical_representative_form
  enriched_family_members
  enriched_representative_form
```

역할:

- canonical family와 enriched family를 병렬로 저장

해석:

- `canonical_*`
  - Unihan-only backbone
  - `variants.family_members` / `variants.representative_form`와 같은 의미
- `enriched_*`
  - canonical edges + supplementary edges를 합친 combined graph 결과

원칙:

- enriched family는 canonical family의 superset 또는 동일 집합이어야 한다.
- backward compatibility 때문에 기존 consumer는 `variants.*`만 읽어도 된다.

### 4.7 `structure`

구조:

```text
structure
  decomposition
  etymology_hint
  etymology_type
  phonetic_component
  semantic_component
```

authority:
- 전부 `MakeMeAHanzi`

설명:

- `decomposition`
  - IDS-like decomposition string
- `etymology_type`
  - 예: `pictophonetic`, `ideographic`, `pictographic`
- `etymology_hint`
  - explanation string
- `phonetic_component`
  - MakeMeAHanzi etymology phonetic field
- `semantic_component`
  - MakeMeAHanzi etymology semantic field

### 4.8 `media`

구조:

```text
media
  stroke_svg_paths
  stroke_medians
```

authority:
- `MakeMeAHanzi`

설명:

- `stroke_svg_paths`
  - stroke별 SVG path string list
- `stroke_medians`
  - stroke별 median polyline coordinate list

`stroke_svg_paths[n]`와 `stroke_medians[n]`는 같은 stroke를 가리킨다.

### 4.9 `references`

구조:

```text
references
  unihan
  kanjidic2
  ehanja
```

이 블록은 source-specific reference subset을 담는다.

예:

- `references.unihan`
  - `rs_unicode`
  - `kangxi`
  - `irg_kangxi`
  - `hanyu`
  - `unihan_core_2020`
- `references.kanjidic2`
  - dictionary refs
  - query codes
  - raw variant refs
  - resolved variant refs
  - unresolved variant refs
  - codepoint refs
  - nanori
- `references.ehanja`
  - `hschool_id`
  - `busu_id`
  - `busu2_id`
  - `hshape`
  - `china_english`
  - `current`
  - `theory`
  - `law`
  - `length`
  - `school_com`

### 4.10 `source_payloads`

구조:

```text
source_payloads
  unihan_raw
  ehanja_raw
  kanjidic2_raw
  makemeahanzi_raw
```

역할:

- canonical flatten 과정에서 잃을 수 있는 source richness를 보존
- conflict / provenance audit를 가능하게 함

원칙:

- canonical field에 채택되지 않은 source 정보도 가능하면 여기에 남긴다.
- 값은 near-raw dict / list / nested structure일 수 있다.

---

## 5. `canonical_variants.jsonl`

이 파일은 **typed variant edge list**다.

각 줄은 directed edge 1개이며, 현재 구조는 다음과 같다.

```text
VariantEdge
  source_name
  relation_scope
  relation
  source_character
  source_codepoint
  target_character
  target_codepoint
```

설명:

- `source_name`
  - `unihan`, `ehanja`, `kanjidic2`
- `relation_scope`
  - `canonical` 또는 `supplementary`
- canonical edge는 Unihan backbone
- supplementary edge는 e-hanja / resolvable KANJIDIC2 evidence
- relation 값은 source-explicit하게 유지된다.
  - canonical 예:
    - `traditional`
    - `simplified`
    - `semantic`
    - `specialized_semantic`
    - `z_variants`
    - `spoofing`
  - supplementary 예:
    - `ehanja_yakja`
    - `ehanja_bonja`
    - `ehanja_simple_china`
    - `ehanja_kanji`
    - `ehanja_dongja`
    - `ehanja_tongja`
    - `kanjidic2_resolved`

이 파일은 graph viewer / variant query / component recomputation의 base artifact다.

---

## 6. `variant_components.jsonl`

이 파일은 **variant connected component summary**다.

각 줄 구조:

```text
VariantComponentRow
  codepoint
  character
  representative_form
  component_size
  family_members
  canonical_representative_form
  canonical_family_members
  enriched_representative_form
  enriched_family_members
```

설명:

- 각 문자마다 자기가 속한 component 정보를 한 줄에 복사해서 들고 있다.
- `representative_form`과 `family_members`는 backward compatibility를 위해 canonical view를 가리킨다.
- `canonical_*`는 명시적인 canonical view다.
- `enriched_*`는 supplementary edges까지 포함한 combined graph 결과다.

즉 `canonical_characters.jsonl`의 `variants.family_members`와 거의 같은 정보를,
component-centric query용으로 따로 분리한 것이라고 볼 수 있다.

---

## 7. `build_summary.json`

이 파일은 build run의 요약 통계다.

현재 구조:

```text
build_summary
  canonical_character_count
  canonical_variant_component_count
  enriched_variant_component_count
  max_canonical_component_size
  max_enriched_component_size
  characters_with_enriched_growth
  sample_codepoints
  source_record_counts
  variant_edge_count
  variant_edge_scope_counts
  supplementary_edge_source_counts
```

역할:

- build sanity check
- coverage audit
- source adapter regression check

---

## 8. SQLite Schema

SQLite 파일 [sinograph_canonical_v1.sqlite](./out/sinograph_canonical_v1.sqlite)는
JSONL의 query-friendly projection이다.

현재 table 목록:

- `characters`
- `character_readings`
- `character_meanings`
- `variant_edges`
- `variant_components`
- `source_presence`
- `source_payloads`
- `character_media`

### 8.1 `characters`

columns:

- `codepoint` TEXT PRIMARY KEY
- `character` TEXT NOT NULL
- `radical` TEXT
- `total_strokes` INTEGER
- `representative_form` TEXT
- `data_json` TEXT NOT NULL

역할:

- canonical character row의 main table
- `data_json` 안에 full canonical record를 JSON으로 보존

### 8.2 `character_readings`

columns:

- `codepoint`
- `character`
- `language`
- `reading_type`
- `value`

역할:

- reading array를 1 row per value로 정규화

현재 구현에서는 `language`와 `reading_type`가 동일 문자열로 들어간다.
예:

- `mandarin`
- `japanese_on`
- `korean_hangul`

### 8.3 `character_meanings`

columns:

- `codepoint`
- `character`
- `language`
- `meaning_type`
- `value`

역할:

- definition array를 1 row per value로 정규화

언어 값은 현재 다음처럼 단순화되어 있다.

- Korean 계열이면 `ko`
- English 계열이면 `en`

### 8.4 `variant_edges`

columns:

- `source_codepoint`
- `source_character`
- `source_name`
- `relation_scope`
- `relation`
- `target_codepoint`
- `target_character`

역할:

- `canonical_variants.jsonl`의 SQLite projection

### 8.5 `variant_components`

columns:

- `codepoint`
- `character`
- `representative_form`
- `component_size`
- `family_members_json`
- `canonical_representative_form`
- `canonical_family_members_json`
- `enriched_representative_form`
- `enriched_family_members_json`

역할:

- component membership lookup
- representative form lookup
- canonical vs enriched family 비교

### 8.6 `source_presence`

columns:

- `codepoint`
- `character`
- `unihan`
- `ehanja`
- `kanjidic2`
- `makemeahanzi`

역할:

- source overlap / 교집합 분석용
- boolean은 현재 `0/1` integer로 저장

### 8.7 `source_payloads`

columns:

- `codepoint`
- `character`
- `source_name`
- `payload_json`

역할:

- source raw payload를 source별로 따로 저장
- provenance / audit / UI debug에 유용

### 8.8 `character_media`

columns:

- `codepoint`
- `character`
- `stroke_index`
- `stroke_svg_path`
- `median_json`

역할:

- stroke media를 stroke 단위 row로 분해
- MakeMeAHanzi graphics를 viewer / renderer에서 다루기 쉽게 만듦

---

## 9. Current Source Responsibility in v1

### Unihan

주 역할:

- canonical `codepoint`
- variant family backbone
- radical / total stroke fallback
- English definition backbone
- Mandarin / Cantonese / Vietnamese backbone

### e-hanja

주 역할:

- Korean hun
- Korean explanation
- Korean-facing reading richness
- Korean educational / normative metadata preservation

### KANJIDIC2

주 역할:

- Japanese `on/kun`
- Korean romanized/hangul fallback
- Japanese dictionary refs

### MakeMeAHanzi

주 역할:

- decomposition
- etymology class / hint
- stroke SVG / medians

---

## 10. Current Limitations

이 DB는 v1이고, 아직 몇 가지 의도적 제약이 있다.

- lexical / word-level merge는 하지 않는다
  - `e-hanja`의 `hWord`, `ftsWord` 등은 제외
- canonical variant semantics는 `Unihan`만 authority로 사용한다
- supplementary variant evidence는 semantic override가 아니다
  - `e-hanja`와 `KANJIDIC2`는 enriched family 확장에만 사용한다
- canonical core는 intentionally narrow하다
  - source richness는 `source_payloads`에 남긴다
- `representative_form`은 heuristic이다
  - 절대적인 philological root를 의미하지 않는다
- 현재 `characters.data_json`에 full JSON을 보존하므로 SQLite 용량은 작지 않을 수 있다

---

## 11. Practical Reading Guide

이 DB를 처음 읽을 때는 보통 이렇게 보면 된다.

1. 문자 1자의 전체 상태를 보고 싶다
   - `characters.data_json`
   - 또는 `canonical_characters.jsonl`

2. 읽기만 빠르게 보고 싶다
   - `character_readings`

3. 뜻만 빠르게 보고 싶다
   - `character_meanings`

4. variant family를 보고 싶다
   - `variant_edges`
   - `variant_components`
   - `canonical_characters.jsonl`의 `variants`는 canonical family
   - `canonical_characters.jsonl`의 `variant_graph`는 canonical vs enriched 비교용

5. stroke graphics를 보고 싶다
   - `character_media`

6. 이 값이 어느 source에서 왔는지 확인하고 싶다
   - `source_presence`
   - `source_payloads`
   - `characters.data_json`

---

## 12. Relationship to the Broader Project

이 canonical DB는 Sinograph Explorer와 OCR 실험의 **중간 backbone** 역할을 한다.

의도된 사용처:

- Sinograph Explorer lookup backend
- variant graph visualization
- uncommon character OCR fallback lookup
- synthetic data generation seed selection
- later v2 source integration의 기반

즉 이 DB는 “최종 단일 진실의 사전”이라기보다,
**여러 source를 provenance-aware하게 통합한 reproducible canonical layer**로 이해하는 것이 맞다.

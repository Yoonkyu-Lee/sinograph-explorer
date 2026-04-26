# Sinograph Canonical DB v1 Schema

이 문서는 `lab3/sinograph_canonical_v1/` 워크스페이스에서 생성할 **통합 Sinograph DB v1**의
canonical record 구조와 ETL 원칙을 정리한다.

핵심 철학은 다음과 같다.

- **intersection + extensions**
  - 여러 source가 공통으로 맞출 수 있는 정보만 canonical core에 둔다.
  - source-specific richness는 버리지 않고 extension / payload로 보존한다.
- **JSONL first**
  - 사람이 diff/inspection 하기 쉬운 JSONL을 canonical intermediate로 삼는다.
  - SQLite는 앱/조회용 파생 산출물이다.
- **source DBs are read-only**
  - `lab3/db_src/` 아래의 원본은 수정하지 않는다.
  - 모든 normalization / merge / export는 `lab3/sinograph_canonical_v1/` 내부에서만 수행한다.

---

## 1. Source Scope

v1의 Core 4 source는 다음과 같다.

- `Unihan`
- `e-hanja`
- `KANJIDIC2`
- `MakeMeAHanzi`

v1에서는 문자 단위(character-level) canonical merge만 수행한다.
단어/성어/복사(複詞) 계열 lexical tables는 v2 이후 별도 layer로 다룬다.

---

## 2. Canonical Identity

각 canonical record의 identity anchor는 다음 두 필드다.

- `character`
  - 실제 한 글자 literal
- `codepoint`
  - `U+XXXX` 형식의 canonical join key

join 우선순위:

1. `codepoint`
2. `character`

원칙:

- source에 `codepoint`가 없고 `character`만 있으면 `ord(character)`로 파생한다.
- multi-character lexical row는 canonical character merge에 넣지 않는다.

---

## 3. Canonical Record Shape

```text
CanonicalCharacter
  character
  codepoint
  source_flags
    unihan
    ehanja
    kanjidic2
    makemeahanzi

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

  variants
    traditional
    simplified
    semantic
    specialized_semantic
    z_variants
    spoofing
    representative_form
    family_members

  supplementary_variants
    ehanja_yakja
    ehanja_bonja
    ehanja_simple_china
    ehanja_kanji
    ehanja_dongja
    ehanja_tongja
    kanjidic2_resolved

  variant_graph
    canonical_family_members
    canonical_representative_form
    enriched_family_members
    enriched_representative_form

  structure
    decomposition
    etymology_type
    etymology_hint
    phonetic_component
    semantic_component

  media
    stroke_svg_paths
    stroke_medians

  references
    unihan
    kanjidic2
    ehanja

  source_payloads
    unihan_raw
    ehanja_raw
    kanjidic2_raw
    makemeahanzi_raw
```

규칙:

- multi-valued field는 가능한 한 모두 list로 유지한다.
- canonical core는 stable cross-source field만 넣는다.
- source-specific detail은 `references` 또는 `source_payloads`에 보존한다.
- missing data는 채우지 않는다. 추측값을 생성하지 않는다.

---

## 4. Source Authority Model

canonical slot별 우선 authority는 다음과 같다.

### Unihan

- canonical `codepoint`
- variant graph backbone
- radical / total strokes fallback
- Unicode-centric dictionary/reference indices

### e-hanja

- Korean hun/eum richness
- Korean explanation
- Korean educational / normative metadata

### KANJIDIC2

- Japanese `on/kun`
- Japanese-centric dictionary references
- auxiliary `pinyin`, `korean_h`, `korean_r`, `vietnam`

### MakeMeAHanzi

- decomposition backbone
- etymology type / hint
- stroke SVG and median geometry

Conflict policy:

- canonical slot은 authority source를 우선한다.
- 충돌하는 값은 버리지 않고 `source_payloads`에 남긴다.
- 의미가 다른 필드를 억지로 하나로 flatten하지 않는다.

---

## 5. Staged ETL Outputs

### 5.1 Staging JSONL

각 source adapter는 먼저 normalized per-character JSONL을 만든다.

권장 파일:

- `staging/unihan.normalized.jsonl`
- `staging/ehanja.normalized.jsonl`
- `staging/kanjidic2.normalized.jsonl`
- `staging/makemeahanzi.normalized.jsonl`

각 staging record는 다음 공통 shape를 갖는다.

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

### 5.2 Canonical Outputs

최종 canonical outputs:

- `out/canonical_characters.jsonl`
- `out/canonical_variants.jsonl`
- `out/variant_components.jsonl`
- `out/build_summary.json`

### 5.3 SQLite Output

- `out/sinograph_canonical_v1.sqlite`

권장 SQLite tables:

- `characters`
- `character_readings`
- `character_meanings`
- `variant_edges`
- `variant_components`
- `source_presence`
- `source_payloads`
- `character_media`

---

## 6. Variant Graph Policy

variant family는 v1.1부터 두 개의 parallel view를 가진다.

- **canonical family**
  - Unihan-only authoritative backbone
- **enriched family**
  - Unihan + supplementary e-hanja + resolvable KANJIDIC2 variant refs

### 6.1 Canonical family

canonical variant semantics는 계속 Unihan만 authority로 사용한다.

사용 relation:

- `kTraditionalVariant`
- `kSimplifiedVariant`
- `kSemanticVariant`
- `kSpecializedSemanticVariant`
- `kZVariant`
- `kSpoofingVariant`

이 관계에서 계산한 결과가 아래 필드에 들어간다.

- `variants.traditional`
- `variants.simplified`
- `variants.semantic`
- `variants.specialized_semantic`
- `variants.z_variants`
- `variants.spoofing`
- `variants.representative_form`
- `variants.family_members`

즉 기존 `variants.*`의 의미는 바뀌지 않는다.

### 6.2 Supplementary family evidence

supplementary source는 canonical relation bucket을 덮어쓰지 않고
별도 확장 필드와 enriched graph에만 참여한다.

#### e-hanja supplementary fields

- `yakja`
- `bonja`
- `simpleChina`
- `kanji`
- `dongja`
- `tongja`

이 값들은 split + single-character filtering 후 codepoint array로 변환되어
`supplementary_variants.*`에 저장된다.

#### KANJIDIC2 supplementary refs

`variant_refs` 중 실제 문자로 resolve 가능한 것만 graph edge로 사용한다.

허용 ref type:

- `jis208`
- `jis212`
- `jis213`
- `ucs`

resolve되지 않는 dictionary/index-only ref는 `source_payloads.kanjidic2_raw`에만 남기고,
variant graph에는 넣지 않는다.

### 6.3 Variant graph outputs

canonical record는 variant family를 두 방식으로 가진다.

- `variants.*`
  - canonical Unihan backbone
- `variant_graph.*`
  - canonical/enriched 두 family view를 병렬 저장

`variant_graph` 구조:

```text
variant_graph
  canonical_family_members
  canonical_representative_form
  enriched_family_members
  enriched_representative_form
```

원칙:

- `variant_graph.canonical_*`는 `variants.family_members` / `variants.representative_form`와 같은 의미다.
- `variant_graph.enriched_*`는 supplementary edges까지 반영한 combined graph 결과다.
- enriched family는 canonical family의 superset 또는 동일 집합이어야 한다.

Representative-form heuristic:

1. traditional/reference-like signal이 더 강한 문자 우선
2. metadata richness가 더 높은 문자 우선
3. 그래도 같으면 lexicographically smallest `codepoint`

`representative_form`은 프로젝트 heuristic이며 philological truth를 주장하지 않는다.

---

## 7. Canonical Fill Policy

### Definitions

- `core.definitions.english`
  - prefer `Unihan.kDefinition`
  - fallback `KANJIDIC2` English meanings
  - fallback `MakeMeAHanzi.definition`

- `core.definitions.korean_explanation`
  - prefer `e-hanja.hRoot.rMeaning`

- `core.definitions.korean_hun`
  - prefer `e-hanja.hRead`

### Readings

- `mandarin`
  - prefer `Unihan.kMandarin`
  - supplement from `KANJIDIC2.pinyin`
  - final fallback `MakeMeAHanzi.pinyin`

- `cantonese`
  - prefer `Unihan.kCantonese`

- `korean_hangul`
  - prefer `e-hanja.hSnd` / `hRoot.rSnd`
  - fallback `KANJIDIC2.korean_h`

- `korean_romanized`
  - prefer `KANJIDIC2.korean_r`

- `japanese_on`
  - prefer `KANJIDIC2.ja_on`
  - fallback `Unihan.kJapaneseOn`

- `japanese_kun`
  - prefer `KANJIDIC2.ja_kun`
  - fallback `Unihan.kJapaneseKun`

- `vietnamese`
  - prefer `Unihan.kVietnamese`
  - fallback `KANJIDIC2.vietnam`

### Structure / Media

- `decomposition`
  - `MakeMeAHanzi.decomposition`
- `etymology_*`
  - `MakeMeAHanzi.etymology`
- `stroke_svg_paths`, `stroke_medians`
  - `MakeMeAHanzi.graphics`

### Radical / Stroke

- `radical`
  - prefer `Unihan` radical extraction
  - fallback `KANJIDIC2 classical radical`
  - fallback `e-hanja` busu id

- `total_strokes`
  - prefer `Unihan.kTotalStrokes`
  - fallback `KANJIDIC2.stroke_count`
  - fallback `e-hanja.hTotal`

---

## 8. Non-Core-4 Sources

다음 source들은 v1 merge에는 직접 넣지 않지만, v2 extension role이 있다.

- `Tongyong Guifan`
  - commonness / tier extension
- `MOE Revised Dict`
  - lexical / definitional Chinese extension
- `CNS11643`
  - Taiwan/CNS mapping extension
- `CEDICT`
  - SC/TC + pinyin + English lexical extension

즉 v1 pipeline은 Core 4만으로 끝까지 동작해야 하며,
나머지는 adapter를 추가해도 깨지지 않는 구조로 설계한다.

# Canonical Schema v2

이 문서는 v2 canonical record 의 정확한 shape 를 고정한다. 설계 근거와 정책은
[`doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md`](../../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md).

## 3-layer 원칙 (요약)

1. **교집합** (≥2 소스가 같은 의미축 공급) → `core.*` 에 단일 값 + `provenance.*` 에 실제 pick source + `core_alternatives.*` 에 다른 소스가 준 값.
2. **합집합 − 교집합** (한 소스 고유) → `source_exclusive.<source>.*` 에 소스명이 경로에 들어감.
3. **Variant 관계** (분류체계가 소스마다 다름) → 소스별 bucket 유지. 같은 관계 쌍을 여러 소스가 지지하면 edge 한 개 + `sources: [...]`.

## Top-level shape

```text
CanonicalCharacter (v2)
  character                str
  codepoint                "U+XXXX"
  source_flags             {unihan, ehanja_online, kanjidic2, makemeahanzi: bool}

  core                     (all list-typed values; empty list is acceptable)
    radical                str | None
    total_strokes          int | None
    definitions
      english              [str]
    readings
      mandarin             [str]
      cantonese            [str]
      korean_hangul        [str]
      japanese_on          [str]
      japanese_kun         [str]
      vietnamese           [str]

  provenance                {field_path: source_name}
    # keys present only if that core field was filled
    # field_path uses dot notation: "readings.mandarin", "total_strokes", "definitions.english"

  core_alternatives         {field_path: {source_name: value}}
    # values from non-picked sources; picked source NOT repeated here.
    # empty / absent for fields where only one source had data.

  variants                  Unihan-only variant backbone (kept at top-level for
                            backward-compat & prominence)
    traditional             ["U+XXXX", ...]
    simplified              ["U+XXXX", ...]
    semantic                ["U+XXXX", ...]
    specialized_semantic    ["U+XXXX", ...]
    z_variants              ["U+XXXX", ...]
    spoofing                ["U+XXXX", ...]
    representative_form     "U+XXXX" | None
    family_members          ["U+XXXX", ...]      # connected component, Unihan canonical

  supplementary_variants    typed per-relation buckets (source-explicit)
    # from e-hanja_online.schoolcom:
    ehanja_yakja            ["U+XXXX", ...]
    ehanja_bonja            ["U+XXXX", ...]
    ehanja_simple           ["U+XXXX", ...]   # schoolcom "simple"
    ehanja_kanji            ["U+XXXX", ...]
    ehanja_dongja           ["U+XXXX", ...]
    ehanja_tongja           ["U+XXXX", ...]
    ehanja_waja             ["U+XXXX", ...]
    ehanja_goja             ["U+XXXX", ...]
    ehanja_sokja            ["U+XXXX", ...]
    ehanja_hDup             ["U+XXXX", ...]
    # from e-hanja_online.detail (distinct dropdown-hover relations):
    ehanja_synonyms         ["U+XXXX", ...]   # 유의
    ehanja_opposites        ["U+XXXX", ...]   # 상대
    ehanja_alt_forms        ["U+XXXX", ...]   # 별자
    # from KANJIDIC2:
    kanjidic2_resolved      ["U+XXXX", ...]

  variant_graph             parallel canonical vs enriched family views
    canonical_family_members        ["U+XXXX", ...]
    canonical_representative_form   "U+XXXX" | None
    enriched_family_members         ["U+XXXX", ...]
    enriched_representative_form    "U+XXXX" | None

  source_exclusive          {source_name: {field: value}}
    unihan
      rs_unicode            [str]
      kangxi                [str]
      irg_kangxi            [str]
      hanyu                 [str]
      unihan_core_2020      [str]
    ehanja_online
      korean_hun            [str]   # getHunum.hRead
      korean_explanation    [{meaning, root_snd, orderA, orderB}, ...]   # getJahae
      classification        {education_level, hanja_grade, name_use}     # detail
      shape                 {representative: {char, gloss}, components: [{char, gloss}, ...]}
      etymology             {type, description}                           # detail
      word_usage            str                                            # detail
      related_words         [{word, reading, word_id}, ...]               # detail
      svg_type              "animated" | "static"                          # from strokes_manifest
      stroke_count_animated int | None
    kanjidic2
      korean_romanized      [str]
      dictionary_refs       {dr_type: [values]}
      query_codes           [{qc_type, value, ...}]
      codepoint_refs        [{cp_type, value}]
      nanori                [str]
      unresolved_variant_refs [{var_type, value}, ...]
    makemeahanzi
      decomposition         str | None     # IDS-like
      etymology_type        str | None
      etymology_hint        str | None
      phonetic_component    str | None
      semantic_component    str | None

  source_payloads           near-raw per-source data for audit
    unihan_raw              dict
    ehanja_online_raw       dict
    kanjidic2_raw           dict
    makemeahanzi_raw        dict
```

## Variant edge record (`canonical_variants.jsonl`)

```text
VariantEdge (v2)
  source_character       str
  source_codepoint       "U+XXXX"
  target_character       str
  target_codepoint       "U+XXXX"
  relation_scope         "canonical" | "supplementary"
  relation               str   # canonical: "traditional" | "simplified" | ... (Unihan)
                               # supplementary: "ehanja_dongja" | ... | "kanjidic2_resolved"
  sources                [str]  # ≥1 source names that support this edge (after merge)
  source_relations       {source_name: relation_name}  # per-source original relation label
  support_count          int    # len(sources). Confidence tier.
```

같은 `(source_cp, target_cp, relation_scope)` 는 merge — scope 가 다르면 별도 row.

## Variant component record (`variant_components.jsonl`)

```text
VariantComponentRow
  codepoint                         "U+XXXX"
  character                         str
  representative_form               "U+XXXX"         # backward compat = canonical_representative_form
  component_size                    int
  family_members                    ["U+XXXX", ...]  # backward compat = canonical
  canonical_representative_form     "U+XXXX"
  canonical_family_members          ["U+XXXX", ...]
  enriched_representative_form      "U+XXXX"
  enriched_family_members           ["U+XXXX", ...]
```

## Staging record (`staging/<source>.normalized.jsonl`)

shared NormalizedSourceRecord:

```text
NormalizedSourceRecord
  source_name                   str
  character                     str
  codepoint                     "U+XXXX"
  radical                       str | None
  total_strokes                 int | None
  readings                      {mandarin, cantonese, korean_hangul, japanese_on, japanese_kun, vietnamese: [str]}
  definitions
    english                     [str]
  variants                      {traditional, simplified, ...: [U+XXXX]}   # Unihan only fills; others empty
  supplementary_variants        {ehanja_*, kanjidic2_resolved: [U+XXXX]}   # source-specific fills
  exclusive                     dict                  # staging-level carrier of source-exclusive data
  source_payload                dict                  # near-raw
```

v1 과 달리 v2 staging 에는 `structure` / `media` / `references` 통합 필드가 **없다**.
그 정보는 전부 `exclusive` dict 안에 담기며, 최종 canonical 의 `source_exclusive.<source>` 로 흘러간다.

## Authority / Fill 정책

섹션 7 (doc/11) 표 그대로. core 는 authority pick + alternatives 보존;
source_exclusive 는 한 소스 고유. 자세한 규칙은 `build_canonical_db_v2.py` 의 projection 주석 참고.

## Variant graph 정책 — Family Graph Exclusion

모든 variant relation 은 `canonical_variants.jsonl` / SQLite `variant_edges` 에
그대로 저장된다. 단 **family component 계산 (connected component on adjacency
graph)** 에 참여하는 relation 은 제한된다.

### Family graph 에 참여하는 relation
- **Unihan canonical** (`relation_scope = "canonical"`): `traditional`, `simplified`, `semantic`, `specialized_semantic`, `z_variants`, `spoofing`
- **e-hanja schoolCom 10종** (`relation_scope = "supplementary"`): `ehanja_dongja`, `ehanja_bonja`, `ehanja_sokja`, `ehanja_yakja`, `ehanja_goja`, `ehanja_waja`, `ehanja_simple`, `ehanja_kanji`, `ehanja_tongja`, `ehanja_hDup`
  - 이 집합은 e-hanja 사이트의 공식 **이체-관계도 (`comsTree.html`)** 이 그리는 관계와 1:1
- **KANJIDIC2**: `kanjidic2_resolved`

### Family graph 에서 **제외** (edge 는 persist, component 계산만 skip)
```python
FAMILY_GRAPH_EXCLUDED_RELATIONS = {
    "ehanja_synonyms",   # 유의자 — 비슷한 뜻의 '다른 글자'. variant 아님.
    "ehanja_opposites",  # 상대자 — 반의자. variant 아님.
    "ehanja_alt_forms",  # 별자 — detail 페이지 dropdown-hover only.
                          # e-hanja 이체-관계도에 안 나타남.
}
```

### 제외 근거
- `synonyms` / `opposites` 는 **"같은 글자의 다른 표기"** 가 아니라 **"다른
  글자지만 의미적 연관"**. family 에 넣으면 semantic-adjacent chain 으로 거대
  cluster 를 만들고 downstream 에서 "이 글자의 이체자는 몇 개" 쿼리가 오염됨.
- `alt_forms` (별자) 는 정의상 variant 이지만 `getSchoolCom` 밖에 있는
  별도 detail 페이지 데이터. e-hanja 공식 이체-관계도에 안 그려지므로 "공식
  variant" 스코프 외로 취급.

### 경계 조건
- 제외된 edge 의 source/target 가 **해당 relation 에서만 연결** 되고 다른
  variant relation 이 없으면: 둘은 서로의 family 에 속하지 않는다 (=
  family size 1 각각).
- 같은 pair 가 이미 다른 relation (예: `ehanja_dongja`) 로도 연결돼 있으면:
  family 는 유지된다 — 제외는 relation 단위이지 edge pair 단위 아님.
- `variant_edges` 테이블 / `canonical_variants.jsonl` 에는 제외 relation 도
  여전히 레코드 존재. 쿼리 시 `relation NOT IN (...)` 으로 필터 가능.

### 실측 효과 (2026-04-19 빌드)
- max enriched component: **15,355 → 60** (256× 축소)
- 殺 (U+6BBA) enriched family: **32 members** — e-hanja 사이트 이체-관계도와 거의 일치
- enriched component 수: 68,096 → 70,712 (거대 cluster 가 의미 단위로 쪼개짐)
- variant edge 총수: 88,740 (변화 없음 — edge 는 그대로)

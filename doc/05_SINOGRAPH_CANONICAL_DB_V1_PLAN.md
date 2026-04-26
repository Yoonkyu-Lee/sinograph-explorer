# Sinograph Canonical DB v1 Plan

## Summary
Build a first integrated Sinograph DB around the **Core 4** sources:

- `Unihan`
- `e-hanja`
- `KANJIDIC2`
- `MakeMeAHanzi`

Use an **intersection + extensions** model:
- keep a small, stable canonical core shared across sources
- preserve source-specific richness in extension blocks instead of flattening everything away

Use **JSONL as the canonical intermediate artifact** and generate **SQLite as the query/app artifact**.

The goal of v1 is not “perfect all-source fusion,” but a clean, reproducible pipeline that:
1. normalizes the same character across sources,
2. preserves provenance,
3. computes a usable variant family backbone,
4. exposes enough integrated metadata for Sinograph Explorer and OCR dataset planning.

## Key Changes

### 1. Define the canonical record model
Use `character` + `codepoint` as the identity anchor, with `codepoint` treated as the canonical join key whenever resolvable.

Canonical record shape for v1:

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

Rules:
- `core` contains only fields we can align reliably.
- `source_payloads` preserves non-canonical richness so information is never lost during normalization.
- multi-valued fields are always stored as arrays, even if many entries have only one value.
- missing data remains missing; do not invent fallback values into canonical fields.

### 2. Freeze source responsibilities
Assign each source a primary authority so merges are deterministic.

Authority model:

- **Unihan**
  - canonical `codepoint`
  - variant graph backbone
  - radical/stroke fallback
  - Unicode-centric dictionary/reference indices
- **e-hanja**
  - Korean explanation
  - Korean hun/eum richness
  - Korean educational/classification metadata
  - Korean-facing variant and normative notes
- **KANJIDIC2**
  - Japanese `on/kun`
  - Japanese-centric dictionary references
  - pinyin / Korean romanized-hangul / Vietnamese as secondary reading supplements
- **MakeMeAHanzi**
  - decomposition backbone
  - etymology class/hint
  - stroke SVG and median geometry

Conflict policy:
- prefer the source assigned above for the canonical slot
- preserve conflicting values in `source_payloads`
- if two sources are both useful but semantically different, do not collapse them into one field; keep one canonical and one extension copy

### 3. Build a staged ETL pipeline
Implement the merge as explicit stages rather than one giant script.

#### Stage A. Source adapters
Create one non-destructive adapter per source that emits normalized per-character JSONL.

Recommended outputs:
- `db_build/staging/unihan.normalized.jsonl`
- `db_build/staging/ehanja.normalized.jsonl`
- `db_build/staging/kanjidic2.normalized.jsonl`
- `db_build/staging/makemeahanzi.normalized.jsonl`

Each adapter should:
- parse the local source format
- convert source-specific field names into a normalized adapter schema
- emit one record per character
- include the original source record under a `raw` or `source_payload` field
- attach `character`, `codepoint` if known, and a `source_name`

Adapter-specific requirements:
- `Unihan`: parse by codepoint first, then derive `character`
- `e-hanja`: use the already-understood `hSchool`-centered join path
- `KANJIDIC2`: one `<character>` => one normalized record
- `MakeMeAHanzi`: join `dictionary.txt` and `graphics.txt` by `character`

#### Stage B. Identity resolution
Create a merge index keyed by:
1. `codepoint` when available
2. otherwise `character`

Rules:
- if a source lacks explicit codepoint but has `character`, compute codepoint from the literal character
- reject malformed multi-character source rows from canonical character merge
- keep lexical resources out of v1 canonical character merge unless they are explicitly single-character rows

This means:
- `e-hanja` character tables join into canonical character records
- `e-hanja` word/idiom tables do not enter v1 canonical character JSONL directly
- `MOE Revised Dict`, `Tongyong`, and other non-Core-4 sources stay out of v1 merge but should remain extension-ready for v2

#### Stage C. Variant graph construction
Build a graph centered on Unihan variant relations.

Use Unihan as the primary edge source:
- `kTraditionalVariant`
- `kSimplifiedVariant`
- `kSemanticVariant`
- `kSpecializedSemanticVariant`
- `kZVariant`
- `kSpoofingVariant`

Graph outputs per record:
- direct variant sets by relation type
- `family_members`
- `representative_form`

Representative-form heuristic for v1:
1. prefer a node with direct `traditional`/reference status if available
2. otherwise prefer the node with the richest combined metadata presence
3. otherwise choose the lexicographically smallest codepoint in the component for determinism

Do not try to claim this is the objectively true “root character”; label it internally as `representative_form`.

#### Stage D. Canonical projection
Merge the staged source records into:
- `canonical_characters.jsonl`
- `canonical_variants.jsonl` or equivalent edge/component artifact if needed for graph views

Canonical fill rules:
- `english` definition:
  - prefer `Unihan.kDefinition`
  - supplement with `KANJIDIC2.meaning` as source extension, not overwrite
- `korean_explanation` / `korean_hun`:
  - prefer `e-hanja`
- `japanese_on`, `japanese_kun`:
  - prefer `KANJIDIC2`
- `mandarin`, `cantonese`:
  - prefer `Unihan`, supplement from `KANJIDIC2` where useful
- `korean_hangul`, `korean_romanized`:
  - prefer `e-hanja` for canonical Korean-facing values; retain KANJIDIC2’s Korean fields as source extensions if different
- `radical`, `total_strokes`:
  - prefer `Unihan`, fall back to source-specific values if missing
- `decomposition`, `etymology`, stroke media:
  - fill from `MakeMeAHanzi`

### 4. Generate SQLite for app/query use
Build a derived SQLite DB from the canonical JSONL.

Recommended SQLite tables:
- `characters`
  - one row per canonical character
- `character_readings`
  - normalized reading rows by language/type
- `character_meanings`
  - meaning rows by language/type
- `variant_edges`
  - typed variant relations
- `variant_components`
  - precomputed component membership / representative form
- `source_presence`
  - source availability flags
- `source_payloads`
  - optional JSON text blob table if keeping raw/source extension data in SQLite
- `character_media`
  - stroke path and median data references or JSON blobs

SQLite is the artifact used by:
- Sinograph Explorer desktop app
- lookup demos
- graph viewer
- OCR dataset selection scripts

JSONL remains the canonical build-layer output because it is easier to diff, inspect, and regenerate.

### 5. Explicitly keep non-Core-4 sources out of v1 merge, but design for v2
Do not block v1 on the remaining DBs. Instead, define how they will fit later.

Planned v2 extension roles:
- `Tongyong Guifan`
  - commonness/tier extension
- `MOE Revised Dict`
  - Chinese lexical/definitional extension
- `CNS11643`
  - Taiwan/CNS mapping extension
- `CEDICT`
  - SC/TC + pinyin + English lexical extension

The pipeline should therefore support future adapters, but v1 implementation should not depend on them.

## Test Plan

### Adapter verification
For each source adapter:
- confirm total record counts are stable against the local source
- confirm key fields are parsed into the normalized staging shape
- spot-check known characters:
  - `學`
  - `学`
  - `斈`
  - one BMP common character
  - one supplementary-plane character if supported by the source

### Identity / merge verification
- confirm the same character from multiple sources merges to one canonical record
- confirm records with only one source still survive
- confirm codepoint derivation from literal character is stable
- confirm no accidental merge of multi-character lexical entries into character records

### Variant verification
- verify `斈`, `學`, `学` family behavior
- verify a more complex family like `鑑 / 鑒 / 鍳 / 鉴 / 𰾫`
- verify each typed Unihan relation lands in the right edge bucket
- verify representative-form selection is deterministic

### Canonical field verification
- confirm canonical Korean explanation is sourced from `e-hanja`
- confirm Japanese `on/kun` are sourced from `KANJIDIC2`
- confirm decomposition and stroke media come from `MakeMeAHanzi`
- confirm raw/source payloads preserve non-canonical information instead of silently dropping it

### SQLite verification
- confirm `characters` row count matches canonical JSONL count
- confirm variant graph tables can reproduce the same connected components as the current viewer logic
- confirm one example app query can retrieve:
  - basic info
  - readings
  - meanings
  - variant family
  - decomposition/media
from SQLite alone

## Assumptions and Defaults
- v1 source scope is fixed to **Core 4** only.
- Canonical artifact strategy is **JSONL first, SQLite derived**.
- Merge philosophy is **intersection + extensions**, not strict intersection and not union-heavy flattening.
- Lexical/word-level tables from `e-hanja` and other sources are out of scope for v1 canonical character merge.
- `representative_form` is a deterministic project heuristic, not a claim of philological truth.
- The plan assumes no attempt to normalize every dictionary nuance into one flat schema; preservation of source richness is a design goal.

# Supplementary Variant Integration v1.1

## Summary
Extend the canonical DB so `e-hanja` and `KANJIDIC2` contribute **supplementary variant evidence** on top of the current Unihan backbone.

The v1.1 design should keep **two parallel family views**:
- **Canonical family**: current Unihan-only `family_members` / `representative_form`
- **Enriched family**: combined graph using Unihan + selected e-hanja relations + resolvable KANJIDIC2 variant refs

This preserves authority and backward compatibility while making the DB more useful for Sinograph Explorer.

## Key Changes

### 1. Keep Unihan as the authoritative variant backbone
Do not change the meaning of the current fields:
- `variants.traditional`
- `variants.simplified`
- `variants.semantic`
- `variants.specialized_semantic`
- `variants.z_variants`
- `variants.spoofing`
- `variants.family_members`
- `variants.representative_form`

These remain **Unihan-derived only**.

### 2. Add explicit supplementary variant blocks
Extend the canonical character record with two new sections.

Add `supplementary_variants`:
- `ehanja_yakja`
- `ehanja_bonja`
- `ehanja_simple_china`
- `ehanja_kanji`
- `ehanja_dongja`
- `ehanja_tongja`
- `kanjidic2_resolved`

Rules:
- Store all values as codepoint arrays.
- Keep source field names visible in the schema instead of over-normalizing them.
- Do not collapse these into Unihan relation names.

Add `variant_graph`:
- `canonical_family_members`
- `canonical_representative_form`
- `enriched_family_members`
- `enriched_representative_form`

Rules:
- `canonical_*` mirrors the existing Unihan-only family.
- `enriched_*` is computed from the combined graph.
- `variants.family_members` and `variants.representative_form` stay as aliases for the canonical Unihan view for backward compatibility.

### 3. e-hanja supplementary edge policy
Parse the following fields from `hSchoolCom` and convert them into single-character codepoint targets:
- `yakja`
- `bonja`
- `simpleChina`
- `kanji`
- `dongja`
- `tongja`

Observed feasibility already supports this:
- all targets are single-character after comma splitting
- counts are large enough to matter, especially `dongja` and `tongja`

Edge policy:
- `yakja` -> supplementary abbreviated-form relation
- `bonja` -> supplementary original-form relation
- `simpleChina` -> supplementary simplified-Chinese relation
- `kanji` -> supplementary Japanese-form relation
- `dongja` -> supplementary same-family/cognate relation
- `tongja` -> supplementary interchangeable-family relation

Implementation rule:
- keep these relation names source-explicit in `supplementary_variants`
- add them into the **enriched graph only**
- do not inject them into Unihan `traditional/simplified/semantic` buckets

### 4. KANJIDIC2 supplementary edge policy
Use `variant_refs` only when they can be resolved back to an actual character through KANJIDIC2 codepoint maps.

Resolved ref types currently worth using:
- `jis208`
- `jis212`
- `jis213`
- `ucs`

Do **not** use unresolved dictionary/index-only refs as graph edges:
- `nelson_c`
- `deroo`
- `oneill`
- `njecd`
- `s_h`
- unresolved `ucs`

Implementation rule:
- build a `cp_type -> value -> character` reverse index from `<codepoint><cp_value>`
- resolve variant refs into characters when possible
- store resolved targets under `supplementary_variants.kanjidic2_resolved`
- retain unresolved refs in `source_payloads.kanjidic2_raw.variant_refs`

Graph rule:
- resolved KANJIDIC2 edges participate in `enriched_family_members`
- they do not alter Unihan canonical relation buckets

### 5. Update artifacts and query surfaces
Update the v1 build pipeline and outputs:
- staging records for `ehanja` should include parsed supplementary variant targets
- staging records for `kanjidic2` should include resolved and unresolved variant refs separately
- `canonical_characters.jsonl` should include the new `supplementary_variants` and `variant_graph` sections
- `canonical_variants.jsonl` should add:
  - `source_name`
  - `relation_scope` (`canonical` or `supplementary`)
  - `relation`
- `variant_components.jsonl` should include both canonical and enriched representative/component views

Update SQLite:
- keep existing `variant_edges` table, but add columns:
  - `source_name`
  - `relation_scope`
- extend `variant_components` with:
  - `canonical_representative_form`
  - `canonical_family_members_json`
  - `enriched_representative_form`
  - `enriched_family_members_json`

Update lookup/reporting scripts:
- `lookup_canonical_db.py` should display both canonical and enriched family summaries
- `analyze_canonical_db.py` should report:
  - canonical vs enriched family counts
  - how many characters gained new family members from supplementary sources
  - source breakdown of supplementary edges

Update docs:
- `canonical_schema_v1.md`
- `CANONICAL_DB_V1_MANUAL.md`
- `README.md`
Document clearly that Unihan remains authoritative while enriched families are practical/project-level expansions.

## Test Plan
Use these checks to validate v1.1:

### Core source-resolution tests
- `e-hanja`:
  - verify all `yakja/bonja/simpleChina/kanji/dongja/tongja` targets are split into single-character codepoints
- `KANJIDIC2`:
  - verify resolvable refs are recovered for `jis208/jis212/jis213/ucs`
  - verify unresolved refs stay in payload only

### Graph behavior tests
- confirm `variants.family_members` is unchanged for current Unihan-only cases
- confirm `variant_graph.enriched_family_members` is a superset or equal to canonical family, never smaller
- verify deterministic `enriched_representative_form`

### Sample character tests
- `學 / 学 / 斈`
  - canonical family should remain the current Unihan family
  - enriched family may grow only if supplementary edges add real members
- `鑑 / 鑒 / 鍳 / 鉴 / 𰾫`
  - verify current family is preserved
  - verify supplementary sources do not incorrectly collapse unrelated characters
- at least one `e-hanja`-rich character with `dongja` / `tongja`
- at least one KANJIDIC2-only variant-ref case

### Acceptance criteria
- backward-compatible consumers can still read current canonical variant fields unchanged
- enriched family data is present and queryable
- unresolved KANJIDIC2 refs are preserved, not silently discarded
- source provenance of supplementary edges is explicit in JSONL and SQLite

## Assumptions
- v1.1 still targets the same Core 4 sources only.
- Unihan remains the only authoritative source for canonical variant semantics.
- `e-hanja` and `KANJIDIC2` are treated as supplementary graph expanders, not semantic overrides.
- The implementer should prefer source-explicit relation names over inventing a single flattened cross-source variant taxonomy.

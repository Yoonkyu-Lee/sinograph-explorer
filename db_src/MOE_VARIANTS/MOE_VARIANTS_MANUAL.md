# 教育部異體字字典 Manual

## Source and Licensing

- **Official title**: `異體字字典`
- **Publisher**: 中華民國教育部
- **Official site**: https://dict.variants.moe.edu.tw
- **Data source**: GitHub mirror `kcwu/moedict-variants` (scraped from official site)
- **Local source file**: [list.txt](./MOE_VARIANTS/list.txt)
- **Local acquisition date**: 2026-04-06
- **Mirror source**: https://github.com/kcwu/moedict-variants

## File Inventory

- [list.txt](./MOE_VARIANTS/list.txt)
  - TSV (tab-separated values)
  - `106,280` rows
  - columns: `id`, `type`, `index`, `character`, `image_path`

## High-Level Data Model

```text
row
  id
  type
  index
  character
  image_path
```

## Column Schema

### `id`

- hierarchical entry identifier
- top-level entries: `A00001`, `B00001`, `C00001`, `N00001`
- sub-entries: `A00001-001`, `A00001-001-1`, etc.
- series prefixes: `A`, `B`, `C`, `N`

### `type`

- observed values: `正`, `附`, or empty
- `正`: standard character (正字) — top-level entry
- `附`: supplementary/appendix entry
- empty: variant sub-entry

### `index`

- numeric variant rank within a sub-group
- `1`, `2`, etc., or empty

### `character`

- Unicode Han character, or empty
- empty when only an image representation exists (no Unicode codepoint assigned)

### `image_path`

- path to scanned/drawn image of the variant form
- format: `/variants/tmp/XXXXX.png`
- present when no Unicode character is available

## Observed Statistics

- total rows: `106,280`
- top-level 正 entries: `29,923`
- 附 entries: `2,000`
- entries with Unicode character: `20,848`
- entries with image only (no Unicode): `85,432`

### Series distribution

| Series | 正 | 附 | Total rows |
|--------|-----|-----|------------|
| A      | 4,808  | 1,279 | 46,221 |
| B      | 6,329  | 549   | 23,623 |
| C      | 18,319 | 163   | 35,417 |
| N      | 468    | 9     | 1,019  |


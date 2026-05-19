//! Response types — the JSON shapes the backend sends to the frontend.
//!
//! Every struct derives `Serialize`, so when a `#[tauri::command]` returns one
//! it is automatically turned into a plain JavaScript object. Fields are `pub`
//! so the query modules (`dict`, `home`) can build them.

use serde::Serialize;

/// A character plus its `U+XXXX` codepoint — the minimal navigable unit.
#[derive(Serialize)]
pub struct NamedChar {
    pub codepoint: String,
    pub character: String,
}

/// Structural facts about a character: radical, stroke counts, IDS breakdown.
#[derive(Serialize)]
pub struct Structure {
    pub radical_idx: Option<i64>,
    pub radical_char: Option<String>,
    pub radical_name: Option<String>,
    pub radical_strokes: Option<i64>,
    pub total_strokes: Option<i64>,
    pub residual_strokes: Option<i64>,
    pub primary_ids: Option<String>,
    pub ids_top_idc: Option<String>,
}

/// Pronunciations in the five non-Korean languages canonical_v3 carries.
#[derive(Serialize, Default)]
pub struct Readings {
    pub mandarin: Vec<String>,
    pub cantonese: Vec<String>,
    pub onyomi: Vec<String>,
    pub kunyomi: Vec<String>,
    pub vietnamese: Vec<String>,
}

/// One Korean 훈음 entry: 자훈 (native gloss) + 독음 (Sino-Korean sound).
#[derive(Serialize)]
pub struct Hunum {
    pub seq: i64,
    pub jahun: Option<String>,
    pub dokeum: String,
}

/// A 자훈 / 독음 pair without the sequence index — used for the daily card.
#[derive(Serialize)]
pub struct HunumPair {
    pub jahun: Option<String>,
    pub dokeum: String,
}

/// Free-text meanings, split by language.
#[derive(Serialize, Default)]
pub struct Meanings {
    pub ko: Vec<String>,
    pub en: Vec<String>,
}

/// One variant relationship edge from a character to a related character.
#[derive(Serialize)]
pub struct VariantEdge {
    pub target_codepoint: String,
    pub target_character: String,
    pub relation: String,
    pub category: String,
}

/// The variant family a character belongs to (group of related forms).
#[derive(Serialize)]
pub struct Family {
    pub size: i64,
    pub representative: String,
    pub members: Vec<NamedChar>,
}

/// 급수 — grade / frequency level across the four standard character sets.
#[derive(Serialize)]
pub struct Grades {
    pub kr_grade: Option<String>,
    pub kr_education: Option<String>,
    pub cn_tonggyong: Option<i64>,
    pub jp_grade: Option<i64>,
    pub jp_freq: Option<i64>,
    pub jp_jlpt: Option<i64>,
    pub unihan_core: Option<String>,
}

/// The full dictionary entry for one character — returned by `lookup`.
#[derive(Serialize)]
pub struct CharacterEntry {
    pub codepoint: String,
    pub character: String,
    pub block: Option<String>,
    pub structure: Structure,
    pub ids_components: Vec<NamedChar>,
    pub readings: Readings,
    pub hunum: Vec<Hunum>,
    pub meanings: Meanings,
    pub variants: Vec<VariantEdge>,
    pub family: Option<Family>,
    pub grades: Option<Grades>,
}

/// One row of a reverse (FTS) search result.
#[derive(Serialize)]
pub struct SearchHit {
    pub codepoint: String,
    pub character: String,
    pub gloss: String,
}

/// A character plus a short gloss — used by the radical-browse list.
#[derive(Serialize)]
pub struct NamedCharGloss {
    pub codepoint: String,
    pub character: String,
    pub gloss: String,
}

/// Database-wide counts shown on the home dashboard.
#[derive(Serialize)]
pub struct DbStats {
    pub characters: i64,
    pub with_reading: i64,
    pub with_meaning: i64,
    pub variant_families: i64,
}

/// One Kangxi radical and how many characters fall under it.
#[derive(Serialize)]
pub struct RadicalInfo {
    pub radical_idx: i64,
    pub character: Option<String>,
    pub name_ko: Option<String>,
    pub strokes: Option<i64>,
    pub char_count: i64,
}

/// The "character of the day" card on the home screen.
#[derive(Serialize)]
pub struct DailyChar {
    pub codepoint: String,
    pub character: String,
    pub gloss: String,
    pub hunum: Option<HunumPair>,
    pub primary_ids: Option<String>,
    pub total_strokes: Option<i64>,
}

/// Everything the home screen needs in one payload.
#[derive(Serialize)]
pub struct HomeData {
    pub stats: DbStats,
    pub radicals: Vec<RadicalInfo>,
    pub daily: Option<DailyChar>,
}

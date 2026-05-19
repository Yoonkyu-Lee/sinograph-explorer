//! Dictionary queries over canonical_v3.sqlite.
//!
//! Holds the read-only database connection (`Db`) and the two core commands:
//! `lookup` (one character -> full entry) and `search` (FTS5 reverse lookup
//! by meaning / reading).

use std::collections::HashSet;
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use tauri::State;

use crate::types::{
    CharacterEntry, Family, Grades, Hunum, Meanings, NamedChar, Readings,
    SearchHit, Structure, VariantEdge,
};
use crate::util::{cp_to_char, fmt_cp, is_idc, normalize_query};

/// Managed state — the single read-only connection to canonical_v3.sqlite,
/// behind a `Mutex` so concurrent commands take turns. Tauri hands a
/// `State<Db>` to any command that asks for one.
pub struct Db(pub Mutex<Connection>);

/// Assemble the full dictionary entry for one codepoint by querying every
/// related table (summary, radical, readings, 훈음, meanings, variants, 급수).
fn build_entry(conn: &Connection, cp: &str) -> Result<CharacterEntry, String> {
    let summary = conn
        .query_row(
            "SELECT character, block, radical_idx, total_strokes, \
             residual_strokes, primary_ids, ids_top_idc \
             FROM character_summary WHERE codepoint = ?1",
            params![cp],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, Option<i64>>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, Option<String>>(6)?,
                ))
            },
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let (character, block, radical_idx, total_strokes, residual_strokes, primary_ids, ids_top_idc) =
        summary.ok_or_else(|| format!("{cp} 는 canonical_v3 universe 에 없습니다."))?;
    let character = match character {
        Some(c) if !c.is_empty() => c,
        _ => cp_to_char(cp),
    };

    // radical reference row
    let (radical_char, radical_name, radical_strokes) = match radical_idx {
        Some(idx) => conn
            .query_row(
                "SELECT char, name_ko, strokes FROM radicals WHERE radical_idx = ?1",
                params![idx],
                |r| {
                    Ok((
                        r.get::<_, Option<String>>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| e.to_string())?
            .unwrap_or((None, None, None)),
        None => (None, None, None),
    };

    // ids component characters that are navigable (present in the universe)
    let mut ids_components = Vec::new();
    if let Some(ids) = &primary_ids {
        let mut member = conn
            .prepare("SELECT 1 FROM characters_ids WHERE codepoint = ?1")
            .map_err(|e| e.to_string())?;
        let mut seen = HashSet::new();
        for ch in ids.chars() {
            if ch.is_ascii() || is_idc(ch) {
                continue;
            }
            let comp_cp = fmt_cp(ch as u32);
            if !seen.insert(comp_cp.clone()) {
                continue;
            }
            let exists = member
                .exists(params![comp_cp])
                .map_err(|e| e.to_string())?;
            if exists {
                ids_components.push(NamedChar {
                    codepoint: comp_cp,
                    character: ch.to_string(),
                });
            }
        }
    }

    // readings (five non-Korean languages)
    let mut readings = Readings::default();
    {
        let mut stmt = conn
            .prepare("SELECT reading_type, value FROM character_readings WHERE codepoint = ?1")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![cp], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (rt, value) = row.map_err(|e| e.to_string())?;
            match rt.as_str() {
                "mandarin" => readings.mandarin.push(value),
                "cantonese" => readings.cantonese.push(value),
                "onyomi" => readings.onyomi.push(value),
                "kunyomi" => readings.kunyomi.push(value),
                "vietnamese" => readings.vietnamese.push(value),
                _ => {}
            }
        }
    }

    // Korean 훈음 (jahun + dokeum pairs)
    let hunum = {
        let mut stmt = conn
            .prepare(
                "SELECT seq, jahun, dokeum FROM character_hunum \
                 WHERE codepoint = ?1 ORDER BY seq",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![cp], |r| {
                Ok(Hunum {
                    seq: r.get(0)?,
                    jahun: r.get(1)?,
                    dokeum: r.get(2)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };

    // meanings (ko / en)
    let mut meanings = Meanings::default();
    {
        let mut stmt = conn
            .prepare("SELECT language, value FROM character_meanings WHERE codepoint = ?1")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![cp], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (lang, value) = row.map_err(|e| e.to_string())?;
            match lang.as_str() {
                "ko" => meanings.ko.push(value),
                "en" => meanings.en.push(value),
                _ => {}
            }
        }
    }

    // variant edges
    let variants = {
        let mut stmt = conn
            .prepare(
                "SELECT target_codepoint, target_character, relation, relation_category \
                 FROM variant_edges WHERE source_codepoint = ?1 \
                 ORDER BY relation_category, relation",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![cp], |r| {
                Ok(VariantEdge {
                    target_codepoint: r.get(0)?,
                    target_character: r.get(1)?,
                    relation: r.get(2)?,
                    category: r.get(3)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };

    // variant family
    let family_row = conn
        .query_row(
            "SELECT component_size, representative, family_members_json \
             FROM variant_family WHERE codepoint = ?1",
            params![cp],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let family = match family_row {
        Some((size, representative, members_json)) if size > 1 => {
            let cps: Vec<String> =
                serde_json::from_str(&members_json).map_err(|e| e.to_string())?;
            let members = cps
                .into_iter()
                .map(|c| NamedChar {
                    character: cp_to_char(&c),
                    codepoint: c,
                })
                .collect();
            Some(Family {
                size,
                representative,
                members,
            })
        }
        _ => None,
    };

    // 급수 — grade / level across the four standard sets
    let grades = conn
        .query_row(
            "SELECT kr_grade, kr_education, cn_tonggyong, jp_grade, \
             jp_freq, jp_jlpt, unihan_core \
             FROM character_grades WHERE codepoint = ?1",
            params![cp],
            |r| {
                Ok(Grades {
                    kr_grade: r.get(0)?,
                    kr_education: r.get(1)?,
                    cn_tonggyong: r.get(2)?,
                    jp_grade: r.get(3)?,
                    jp_freq: r.get(4)?,
                    jp_jlpt: r.get(5)?,
                    unihan_core: r.get(6)?,
                })
            },
        )
        .optional()
        .map_err(|e| e.to_string())?;

    Ok(CharacterEntry {
        codepoint: cp.to_string(),
        character,
        block,
        structure: Structure {
            radical_idx,
            radical_char,
            radical_name,
            radical_strokes,
            total_strokes,
            residual_strokes,
            primary_ids,
            ids_top_idc,
        },
        ids_components,
        readings,
        hunum,
        meanings,
        variants,
        family,
        grades,
    })
}

/// Command (frontend calls it via `invoke("lookup", ...)`): resolve a query to
/// a codepoint and return that character's full dictionary entry.
#[tauri::command]
pub fn lookup(db: State<Db>, query: String) -> Result<CharacterEntry, String> {
    let cp = normalize_query(&query)?;
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    build_entry(&conn, &cp)
}

/// Command: reverse search over the FTS5 index — finds characters by meaning
/// or reading, returning up to `limit` hits each with a short gloss.
#[tauri::command]
pub fn search(db: State<Db>, query: String, limit: i64) -> Result<Vec<SearchHit>, String> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    // quote the query so FTS5 treats it as a literal phrase, not syntax.
    let match_expr = format!("\"{}\"", q.replace('"', " "));
    let limit = if (1..=200).contains(&limit) { limit } else { 50 };

    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let pairs: Vec<(String, String)> = {
        let mut stmt = conn
            .prepare(
                "SELECT codepoint, hanja FROM fts_search \
                 WHERE fts_search MATCH ?1 LIMIT ?2",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![match_expr, limit], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };

    let mut hits = Vec::with_capacity(pairs.len());
    for (codepoint, character) in pairs {
        let gloss: Option<String> = conn
            .query_row(
                "SELECT value FROM character_meanings \
                 WHERE codepoint = ?1 AND language = 'ko' LIMIT 1",
                params![codepoint],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        hits.push(SearchHit {
            codepoint,
            character,
            gloss: gloss.unwrap_or_default(),
        });
    }
    Ok(hits)
}

//! Home screen data — database statistics, the radical index, and a stable
//! "character of the day".

use rusqlite::{params, Connection, OptionalExtension};
use tauri::State;

use crate::dict::Db;
use crate::types::{
    DailyChar, DbStats, HomeData, HunumPair, NamedCharGloss, RadicalInfo,
};
use crate::util::{cp_to_char, day_number};

/// Build the daily-character card for a given codepoint.
fn daily_char(conn: &Connection, cp: &str) -> Result<DailyChar, String> {
    let (character, primary_ids, total_strokes) = conn
        .query_row(
            "SELECT character, primary_ids, total_strokes \
             FROM character_summary WHERE codepoint = ?1",
            params![cp],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                ))
            },
        )
        .map_err(|e| e.to_string())?;
    let gloss: String = conn
        .query_row(
            "SELECT value FROM character_meanings \
             WHERE codepoint = ?1 AND language = 'ko' LIMIT 1",
            params![cp],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let hunum: Option<HunumPair> = conn
        .query_row(
            "SELECT jahun, dokeum FROM character_hunum \
             WHERE codepoint = ?1 ORDER BY seq LIMIT 1",
            params![cp],
            |r| {
                Ok(HunumPair {
                    jahun: r.get(0)?,
                    dokeum: r.get(1)?,
                })
            },
        )
        .optional()
        .map_err(|e| e.to_string())?;
    Ok(DailyChar {
        codepoint: cp.to_string(),
        character: character.filter(|c| !c.is_empty()).unwrap_or_else(|| cp_to_char(cp)),
        gloss,
        hunum,
        primary_ids,
        total_strokes,
    })
}

/// Command: everything the home dashboard needs — stats, radical index, and
/// the day's featured character.
#[tauri::command]
pub fn home_data(db: State<Db>) -> Result<HomeData, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let count = |sql: &str| -> Result<i64, String> {
        conn.query_row(sql, [], |r| r.get(0)).map_err(|e| e.to_string())
    };

    let stats = DbStats {
        characters: count("SELECT count(*) FROM characters_ids")?,
        with_reading: count("SELECT count(DISTINCT codepoint) FROM character_readings")?,
        with_meaning: count("SELECT count(DISTINCT codepoint) FROM character_meanings")?,
        variant_families: count(
            "SELECT count(*) FROM variant_family WHERE component_size > 1",
        )?,
    };

    let radicals = {
        let mut stmt = conn
            .prepare(
                "SELECT r.radical_idx, r.char, r.name_ko, r.strokes, \
                 (SELECT count(*) FROM characters_structure s \
                  WHERE s.radical_idx = r.radical_idx) \
                 FROM radicals r ORDER BY r.radical_idx",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| {
                Ok(RadicalInfo {
                    radical_idx: r.get(0)?,
                    character: r.get(1)?,
                    name_ko: r.get(2)?,
                    strokes: r.get(3)?,
                    char_count: r.get(4)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };

    // daily character — stable per day, drawn from CJK Unified characters
    // that carry a Korean meaning (so it is a "normal" dictionary entry).
    const DAILY_POOL: &str = "SELECT DISTINCT m.codepoint FROM character_meanings m \
         JOIN characters_core c ON m.codepoint = c.codepoint \
         WHERE m.language = 'ko' AND c.block = 'CJK Unified'";
    let pool: i64 = count(&format!("SELECT count(*) FROM ({DAILY_POOL})"))?;
    let daily = if pool > 0 {
        let offset = day_number().rem_euclid(pool);
        let cp: String = conn
            .query_row(
                &format!("{DAILY_POOL} ORDER BY m.codepoint LIMIT 1 OFFSET ?1"),
                params![offset],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        Some(daily_char(&conn, &cp)?)
    } else {
        None
    };

    Ok(HomeData {
        stats,
        radicals,
        daily,
    })
}

/// Command: list the characters that fall under one Kangxi radical, ordered
/// by stroke count, each with a short gloss.
#[tauri::command]
pub fn radical_chars(
    db: State<Db>,
    radical_idx: i64,
    limit: i64,
) -> Result<Vec<NamedCharGloss>, String> {
    let limit = if (1..=2000).contains(&limit) { limit } else { 500 };
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT s.codepoint, c.character, \
             (SELECT value FROM character_meanings m \
              WHERE m.codepoint = s.codepoint AND m.language = 'ko' LIMIT 1) \
             FROM characters_structure s \
             JOIN characters_core c ON s.codepoint = c.codepoint \
             WHERE s.radical_idx = ?1 \
             ORDER BY s.total_strokes, s.codepoint LIMIT ?2",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![radical_idx, limit], |r| {
            Ok(NamedCharGloss {
                codepoint: r.get(0)?,
                character: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                gloss: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

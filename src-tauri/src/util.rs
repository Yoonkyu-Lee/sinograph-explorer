//! Small shared helpers — codepoint formatting and lookup-query normalization.

/// Format a Unicode scalar value as a `U+XXXX` string.
pub fn fmt_cp(value: u32) -> String {
    format!("U+{value:04X}")
}

/// The literal character for a `U+XXXX` codepoint string (empty if invalid).
pub fn cp_to_char(cp: &str) -> String {
    cp.strip_prefix("U+")
        .and_then(|h| u32::from_str_radix(h, 16).ok())
        .and_then(char::from_u32)
        .map(|c| c.to_string())
        .unwrap_or_default()
}

/// Normalize a lookup query into a `U+XXXX` codepoint string. Accepts one
/// literal character, a `U+XXXX` string, or bare hex.
pub fn normalize_query(q: &str) -> Result<String, String> {
    let q = q.trim();
    if q.is_empty() {
        return Err("입력이 비어 있습니다.".into());
    }
    let upper = q.to_uppercase();
    let hex = if let Some(rest) = upper.strip_prefix("U+") {
        Some(rest.to_string())
    } else if upper.len() >= 4 && upper.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(upper.clone())
    } else {
        None
    };
    if let Some(h) = hex {
        let value = u32::from_str_radix(&h, 16)
            .map_err(|_| format!("잘못된 코드포인트: {q}"))?;
        return Ok(fmt_cp(value));
    }
    let ch = q.chars().next().expect("non-empty checked above");
    Ok(fmt_cp(ch as u32))
}

/// Is this char an Ideographic Description Character — a structure operator
/// (⿰ ⿱ …), not an actual component?
pub fn is_idc(c: char) -> bool {
    matches!(c as u32, 0x2FF0..=0x2FFF | 0x31EF)
}

/// Whole days since the Unix epoch — used to pick a stable "daily" character.
pub fn day_number() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| (d.as_secs() / 86_400) as i64)
        .unwrap_or(0)
}

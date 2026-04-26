use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

const WANTED_FIELDS: &[&str] = &[
    "kDefinition",
    "kMandarin",
    "kCantonese",
    "kJapanese",
    "kJapaneseOn",
    "kJapaneseKun",
    "kKorean",
    "kTotalStrokes",
    "kRSUnicode",
    "kTraditionalVariant",
    "kSimplifiedVariant",
    "kSemanticVariant",
    "kSpecializedSemanticVariant",
    "kSpoofingVariant",
    "kZVariant",
    "kIRGKangXi",
    "kKangXi",
    "kHanYu",
    "kUnihanCore2020",
];

const VARIANT_FIELDS: &[&str] = &[
    "kTraditionalVariant",
    "kSimplifiedVariant",
    "kSemanticVariant",
    "kSpecializedSemanticVariant",
    "kSpoofingVariant",
    "kZVariant",
];

static UNIHAN_DATA: OnceLock<Result<UnihanData, String>> = OnceLock::new();

#[derive(Clone, Default)]
struct Entry {
    character: String,
    fields: BTreeMap<String, String>,
}

type VariantGraph = BTreeMap<String, BTreeMap<String, BTreeSet<String>>>;

struct UnihanData {
    db: BTreeMap<String, Entry>,
    graph: VariantGraph,
}

#[derive(Serialize)]
struct BasicInfo {
    definition: String,
    mandarin: String,
    cantonese: String,
    japanese: String,
    japanese_on: String,
    japanese_kun: String,
    korean: String,
    total_strokes: String,
    radical_stroke: String,
    unihan_core: String,
}

#[derive(Serialize)]
struct DictionaryReferences {
    kangxi: String,
    irg_kangxi: String,
    hanyu: String,
}

#[derive(Serialize)]
struct VariantRelation {
    field: String,
    raw_value: String,
    linked: Vec<LinkedCharacter>,
}

#[derive(Serialize)]
struct LinkedCharacter {
    character: String,
    codepoint: String,
    definition: String,
    mandarin: String,
}

#[derive(Serialize)]
struct DiscoveredEdge {
    source_character: String,
    source_codepoint: String,
    relation: String,
    target_character: String,
    target_codepoint: String,
}

#[derive(Serialize)]
struct LookupResponse {
    character: String,
    codepoint: String,
    status: String,
    basic_info: Option<BasicInfo>,
    variant_relations: Vec<VariantRelation>,
    dictionary_references: Option<DictionaryReferences>,
    component_visit_order: Vec<String>,
    component_nodes: Vec<LinkedCharacter>,
    discovered_edges: Vec<DiscoveredEdge>,
    note: String,
}

fn field_or_none(entry: &Entry, key: &str) -> String {
    entry.fields
        .get(key)
        .cloned()
        .unwrap_or_else(|| "(none)".to_string())
}

fn codepoint_to_character(cp: &str) -> Option<String> {
    let hex = cp.strip_prefix("U+")?;
    let value = u32::from_str_radix(hex, 16).ok()?;
    char::from_u32(value).map(|ch| ch.to_string())
}

fn extract_codepoints(value: &str) -> Vec<String> {
    value.split_whitespace()
        .filter_map(|token| {
            let candidate = token.split('<').next().unwrap_or(token);
            if candidate.starts_with("U+")
                && (candidate.len() == 6 || candidate.len() == 7 || candidate.len() == 8)
                && candidate.chars().skip(2).all(|ch| ch.is_ascii_hexdigit())
            {
                Some(candidate.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn unihan_dir_candidates() -> [PathBuf; 2] {
    [
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("db_src")
            .join("Unihan")
            .join("Unihan_txt"),
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("Unihan")
            .join("Unihan_txt"),
    ]
}

fn load_unihan_data() -> Result<UnihanData, String> {
    let unihan_dir = unihan_dir_candidates()
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| "Could not find the Unihan directory for the Tauri backend.".to_string())?;

    let wanted: HashSet<&str> = WANTED_FIELDS.iter().copied().collect();
    let mut db: BTreeMap<String, Entry> = BTreeMap::new();

    let mut files: Vec<_> = fs::read_dir(&unihan_dir)
        .map_err(|e| format!("Failed to read Unihan dir: {e}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("Unihan_") && name.ends_with(".txt"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    for path in files {
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut parts = line.splitn(3, '\t');
            let codepoint = match parts.next() {
                Some(value) => value,
                None => continue,
            };
            let field = match parts.next() {
                Some(value) => value,
                None => continue,
            };
            let value = match parts.next() {
                Some(value) => value,
                None => continue,
            };

            if !wanted.contains(field) {
                continue;
            }

            let entry = db.entry(codepoint.to_string()).or_default();
            entry.character = codepoint_to_character(codepoint).unwrap_or_else(|| "?".to_string());
            entry.fields.insert(field.to_string(), value.to_string());
        }
    }

    let mut graph: VariantGraph = BTreeMap::new();
    for (cp, entry) in &db {
        for field in VARIANT_FIELDS {
            if let Some(raw_value) = entry.fields.get(*field) {
                for target_cp in extract_codepoints(raw_value) {
                    graph
                        .entry(cp.clone())
                        .or_default()
                        .entry((*field).to_string())
                        .or_default()
                        .insert(target_cp.clone());

                    graph
                        .entry(target_cp.clone())
                        .or_default()
                        .entry((*field).to_string())
                        .or_default()
                        .insert(cp.clone());
                }
            }
        }
    }

    Ok(UnihanData { db, graph })
}

fn get_unihan_data() -> Result<&'static UnihanData, String> {
    match UNIHAN_DATA.get_or_init(load_unihan_data) {
        Ok(data) => Ok(data),
        Err(err) => Err(err.clone()),
    }
}

fn linked_character_from_cp(cp: &str, db: &BTreeMap<String, Entry>) -> LinkedCharacter {
    let entry = db.get(cp);
    LinkedCharacter {
        character: entry
            .map(|e| e.character.clone())
            .or_else(|| codepoint_to_character(cp))
            .unwrap_or_else(|| "?".to_string()),
        codepoint: cp.to_string(),
        definition: entry
            .map(|e| field_or_none(e, "kDefinition"))
            .unwrap_or_else(|| "(none)".to_string()),
        mandarin: entry
            .map(|e| field_or_none(e, "kMandarin"))
            .unwrap_or_else(|| "(none)".to_string()),
    }
}

fn traverse_variant_component(start_cp: &str, graph: &VariantGraph) -> Vec<String> {
    if !graph.contains_key(start_cp) {
        return vec![start_cp.to_string()];
    }

    let mut queue = VecDeque::from([start_cp.to_string()]);
    let mut visited: BTreeSet<String> = BTreeSet::from([start_cp.to_string()]);
    let mut visit_order = vec![start_cp.to_string()];

    while let Some(current) = queue.pop_front() {
        if let Some(relations) = graph.get(&current) {
            for neighbors in relations.values() {
                for neighbor in neighbors {
                    if visited.insert(neighbor.clone()) {
                        queue.push_back(neighbor.clone());
                        visit_order.push(neighbor.clone());
                    }
                }
            }
        }
    }

    visit_order
}

fn collect_directed_component_edges(
    component_nodes: &BTreeSet<String>,
    db: &BTreeMap<String, Entry>,
) -> Vec<(String, String, String)> {
    let mut edge_keys: BTreeSet<(String, String, String)> = BTreeSet::new();

    for cp in component_nodes {
        if let Some(entry) = db.get(cp) {
            for field in VARIANT_FIELDS {
                if let Some(raw_value) = entry.fields.get(*field) {
                    for target_cp in extract_codepoints(raw_value) {
                        if component_nodes.contains(&target_cp) {
                            edge_keys.insert((cp.clone(), target_cp, (*field).to_string()));
                        }
                    }
                }
            }
        }
    }

    edge_keys.into_iter().collect()
}

#[tauri::command]
fn lookup_character(character: String) -> Result<LookupResponse, String> {
    let mut chars = character.trim().chars();
    let ch = chars
        .next()
        .ok_or_else(|| "Please provide one character.".to_string())?;

    if chars.next().is_some() {
        return Err("Please provide exactly one character.".to_string());
    }

    let cp = format!("U+{:04X}", ch as u32);
    let data = get_unihan_data()?;

    let Some(entry) = data.db.get(&cp) else {
        return Ok(LookupResponse {
            character: ch.to_string(),
            codepoint: cp,
            status: "Not found in loaded Unihan subset".to_string(),
            basic_info: None,
            variant_relations: Vec::new(),
            dictionary_references: None,
            component_visit_order: Vec::new(),
            component_nodes: Vec::new(),
            discovered_edges: Vec::new(),
            note: "The backend is connected, but this codepoint was not found in the loaded Unihan fields.".to_string(),
        });
    };

    let basic_info = BasicInfo {
        definition: field_or_none(entry, "kDefinition"),
        mandarin: field_or_none(entry, "kMandarin"),
        cantonese: field_or_none(entry, "kCantonese"),
        japanese: field_or_none(entry, "kJapanese"),
        japanese_on: field_or_none(entry, "kJapaneseOn"),
        japanese_kun: field_or_none(entry, "kJapaneseKun"),
        korean: field_or_none(entry, "kKorean"),
        total_strokes: field_or_none(entry, "kTotalStrokes"),
        radical_stroke: field_or_none(entry, "kRSUnicode"),
        unihan_core: field_or_none(entry, "kUnihanCore2020"),
    };

    let dictionary_references = DictionaryReferences {
        kangxi: field_or_none(entry, "kKangXi"),
        irg_kangxi: field_or_none(entry, "kIRGKangXi"),
        hanyu: field_or_none(entry, "kHanYu"),
    };

    let mut variant_relations = Vec::new();
    for field in VARIANT_FIELDS {
        let raw_value = field_or_none(entry, field);
        let linked = if raw_value == "(none)" {
            Vec::new()
        } else {
            extract_codepoints(&raw_value)
                .into_iter()
                .map(|target_cp| linked_character_from_cp(&target_cp, &data.db))
                .collect()
        };

        variant_relations.push(VariantRelation {
            field: (*field).to_string(),
            raw_value,
            linked,
        });
    }

    let component_visit_order = traverse_variant_component(&cp, &data.graph);
    let component_node_set: BTreeSet<String> = component_visit_order.iter().cloned().collect();
    let edges = collect_directed_component_edges(&component_node_set, &data.db);
    let component_nodes = component_visit_order
        .iter()
        .map(|cp| linked_character_from_cp(cp, &data.db))
        .collect();

    let discovered_edges = edges
        .into_iter()
        .map(|(source, target, relation)| DiscoveredEdge {
            source_character: linked_character_from_cp(&source, &data.db).character,
            source_codepoint: source,
            relation,
            target_character: linked_character_from_cp(&target, &data.db).character,
            target_codepoint: target,
        })
        .collect();

    Ok(LookupResponse {
        character: ch.to_string(),
        codepoint: cp,
        status: "Found in Unihan".to_string(),
        basic_info: Some(basic_info),
        variant_relations,
        dictionary_references: Some(dictionary_references),
        component_visit_order,
        component_nodes,
        discovered_edges,
        note: "This backend now mirrors the Python demo flow: parse selected Unihan fields, build an undirected variant graph, and run BFS from the recognized character.".to_string(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![lookup_character])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

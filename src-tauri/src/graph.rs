//! Variant relationship graph — builds the node/edge data the frontend's
//! cytoscape view renders for one character's variant family.

use std::collections::HashSet;

use rusqlite::{params, OptionalExtension};
use serde::Serialize;
use tauri::State;

use crate::dict::Db;
use crate::util::{cp_to_char, normalize_query};

/// One node in the variant graph. `focus` marks the queried character.
#[derive(Serialize)]
pub struct GraphNode {
    pub codepoint: String,
    pub character: String,
    pub focus: bool,
}

/// One (undirected) edge between two family members.
#[derive(Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub category: String,
}

/// The whole graph payload returned to the frontend.
#[derive(Serialize)]
pub struct VariantGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Command: the variant family of one character as a graph — family members
/// are nodes, and `variant_edges` among those members are the (undirected,
/// de-duplicated) edges.
#[tauri::command]
pub fn variant_graph(db: State<Db>, query: String) -> Result<VariantGraph, String> {
    let cp = normalize_query(&query)?;
    let conn = db.0.lock().map_err(|e| e.to_string())?;

    let members_json: Option<String> = conn
        .query_row(
            "SELECT family_members_json FROM variant_family WHERE codepoint = ?1",
            params![cp],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let members: Vec<String> = match members_json {
        Some(j) => serde_json::from_str(&j).map_err(|e| e.to_string())?,
        None => vec![cp.clone()],
    };
    let member_set: HashSet<&str> = members.iter().map(String::as_str).collect();

    let nodes = members
        .iter()
        .map(|m| GraphNode {
            character: cp_to_char(m),
            focus: *m == cp,
            codepoint: m.clone(),
        })
        .collect();

    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT source_codepoint, target_codepoint, relation, \
                 relation_category FROM variant_edges WHERE source_codepoint = ?1",
            )
            .map_err(|e| e.to_string())?;
        for m in &members {
            let rows = stmt
                .query_map(params![m], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                })
                .map_err(|e| e.to_string())?;
            for row in rows {
                let (s, t, relation, category) = row.map_err(|e| e.to_string())?;
                if !member_set.contains(t.as_str()) {
                    continue;
                }
                // de-duplicate to one undirected edge per (pair, relation)
                let pair = if s < t {
                    (s.clone(), t.clone(), relation.clone())
                } else {
                    (t.clone(), s.clone(), relation.clone())
                };
                if seen.insert(pair) {
                    edges.push(GraphEdge {
                        source: s,
                        target: t,
                        relation,
                        category,
                    });
                }
            }
        }
    }

    Ok(VariantGraph { nodes, edges })
}

//! Sinograph Dictionary — Tauri backend.
//!
//! A hanzi dictionary desktop app over `canonical_v3.sqlite` (bundled as a
//! Tauri resource) with on-screen hanzi recognition (the SCER model).
//!
//! The backend is split by role:
//!   - `types`     — JSON response shapes sent to the frontend
//!   - `util`      — codepoint / query helpers
//!   - `dict`      — the DB connection state + `lookup` / `search`
//!   - `graph`     — the variant relationship graph
//!   - `home`      — home-screen data (stats, radicals, daily character)
//!   - `recognize` — SCER model inference, screen capture, recognition commands
//!   - `overlay`   — the capture-overlay window
//!
//! `run()` below wires it all together: it opens the database, loads the
//! recognition model, registers the global hotkey, and lists every command
//! the frontend may call.

use std::sync::{Arc, Mutex};

use rusqlite::{Connection, OpenFlags};
use tauri::Manager;

mod dict;
mod graph;
mod home;
mod overlay;
mod types;
mod util;
pub mod recognize;

use dict::Db;
use recognize::Recognizer;

/// Application entry point — builds and runs the Tauri app.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // global hotkey plugin — Ctrl+Shift+H is handled entirely in Rust
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    use tauri_plugin_global_shortcut::ShortcutState;
                    if event.state() == ShortcutState::Pressed {
                        let app = app.clone();
                        tauri::async_runtime::spawn(async move {
                            if let Err(e) = overlay::open_overlay(app).await {
                                eprintln!("[hotkey] open overlay failed: {e}");
                            }
                        });
                    }
                })
                .build(),
        )
        // setup runs once at startup — load resources into managed state
        .setup(|app| {
            // resolve a bundled resource file to an absolute path
            let resource = |name: &str| {
                app.path()
                    .resolve(
                        format!("resources/{name}"),
                        tauri::path::BaseDirectory::Resource,
                    )
                    .unwrap_or_else(|e| panic!("resolve resource {name}: {e}"))
            };

            // open canonical_v3.sqlite read-only and share it as state
            let db_path = resource("canonical_v3.sqlite");
            let conn = Connection::open_with_flags(
                &db_path,
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
            .unwrap_or_else(|e| {
                panic!("failed to open {}: {e}", db_path.display())
            });
            app.manage(Db(Mutex::new(conn)));

            // load the SCER recognition engine and share it as state (doc/39)
            let recognizer = Recognizer::load(
                &resource("scer_v4.onnx"),
                &resource("scer_anchor_db_v20.npy"),
                &resource("class_index.json"),
            )
            .unwrap_or_else(|e| panic!("failed to load SCER recognizer: {e}"));
            app.manage(Arc::new(recognizer));

            // global hotkey — Ctrl+Shift+H opens the capture overlay
            {
                use tauri_plugin_global_shortcut::{
                    Code, GlobalShortcutExt, Modifiers, Shortcut,
                };
                let hotkey = Shortcut::new(
                    Some(Modifiers::CONTROL | Modifiers::SHIFT),
                    Code::KeyH,
                );
                if let Err(e) = app.global_shortcut().register(hotkey) {
                    eprintln!("[setup] hotkey register failed: {e}");
                }
            }

            Ok(())
        })
        // every command the frontend may invoke
        .invoke_handler(tauri::generate_handler![
            dict::lookup,
            dict::search,
            graph::variant_graph,
            home::home_data,
            home::radical_chars,
            recognize::recognize_image_file,
            recognize::recognize_under_cursor,
            overlay::open_capture_overlay,
            overlay::close_capture_overlay,
            overlay::focus_main_with_lookup
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

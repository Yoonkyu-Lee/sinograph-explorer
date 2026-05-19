# Sinograph Explorer

A Tauri 2 desktop hanzi dictionary over the **canonical_v3** character
database, with on-screen hanzi recognition powered by an embedded ONNX model
(SCER — Structure-Conditioned Embedding Recognition).

> The directory / repo name keeps the early project name. The Tauri product
> name is "Sinograph Dictionary".

## Features

- **Direct lookup** — type one hanzi or a `U+XXXX` code → an entry with five
  panels: structure (IDS decomposition, radical, strokes), readings (Korean
  훈음 + Mandarin / Cantonese / Japanese / Vietnamese), meanings, variants,
  and grade levels.
- **Reverse search (FTS5)** — type a meaning or reading word → matching hanzi.
- **Home dashboard** — database stats, a daily character, recently viewed,
  and a 214-radical browse grid.
- **Variant graph** — the variant family of a character as a live
  force-directed graph (cytoscape); tap a node to navigate to it.
- **On-screen recognition** — press `Ctrl+Shift+H` (or the toolbar button),
  point the rectangle at any hanzi on screen, and pick from the recognized
  candidates to open its entry.
- **Click navigation** everywhere — IDS components, radicals, variants,
  family members, and search hits are all clickable, with a back stack.

## Project layout

```
index.html / overlay.html        the two webview entry points
src/                             frontend (Vite + vanilla JS)
  main.js                        orchestrator — wiring + event delegation
  util.js                        shared constants + stateless helpers
  home.js / entry.js             home screen / entry detail rendering
  graph.js                       cytoscape variant graph
  nav.js                         navigation state + actions
  overlay.js                     capture-overlay UI
  styles.css / overlay.css
src-tauri/                       Rust backend (Tauri 2)
  src/lib.rs                     app setup — wires the modules together
  src/types.rs                   JSON response shapes
  src/util.rs                    codepoint / query helpers
  src/dict.rs                    DB connection + lookup / search
  src/graph.rs                   variant relationship graph
  src/home.rs                    home-screen data
  src/recognize.rs               SCER ONNX inference + screen capture
  src/overlay.rs                 capture-overlay window management
  resources/                     bundled data files (see below)
```

## Bundled resources

`src-tauri/resources/` holds four files bundled into the app at build time:

| file | size | tracked in git | source |
|---|---|---|---|
| `scer_v4.onnx` | ~43 MB | yes | SCER model export |
| `class_index.json` | ~2 MB | yes | class index → codepoint map |
| `canonical_v3.sqlite` | ~99 MB | no (gitignored) | the canonical_v3 build pipeline |
| `scer_anchor_db_v20.npy` | ~48 MB | no (gitignored) | SCER anchor DB export |

The two large files are not committed. To build from a fresh clone, obtain
them from the data/ML pipeline that produced them and place them in
`src-tauri/resources/`.

## Develop / build

```powershell
npm install
npm run tauri dev      # run in development
npm run tauri build    # produce an installer
```

Requires the Rust toolchain and the Tauri 2 prerequisites. The SQLite engine
(with FTS5) is compiled in via `rusqlite`'s `bundled` feature; recognition
runs in pure Rust via the `tract-onnx` crate.

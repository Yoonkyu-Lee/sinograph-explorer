# Unihan Graph Viewer

Minimal Tauri 2.0 desktop scaffold for exploring Unihan metadata and visualizing variant graphs.

## Dev setup

From the project root:

```powershell
cd .\lab3\unihan_graph_viewer
npm.cmd install
npm.cmd run tauri dev
```

## Current state

- Vanilla Vite frontend
- Tauri 2.0 Rust backend
- Simple test command: `lookup_character`
- Placeholder UI for future graph visualization

## Suggested next steps

1. Keep `lab3/db_src/Unihan/unihan_lookup_demo.py` and the Rust backend behavior in sync.
2. Add a graph library such as Cytoscape.js.
3. Render node/edge relationships in the right-hand panel.

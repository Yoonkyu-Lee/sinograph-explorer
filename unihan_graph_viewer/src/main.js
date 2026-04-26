import "./styles.css";
import { invoke } from "@tauri-apps/api/core";
import cytoscape from "cytoscape";
import nodeHtmlLabel from "cytoscape-node-html-label";

const input = document.querySelector("#character-input");
const button = document.querySelector("#lookup-button");
const summaryPanel = document.querySelector("#summary-panel");
const relationsPanel = document.querySelector("#relations-panel");
const graphPanel = document.querySelector("#graph-panel");
const rawJson = document.querySelector("#raw-json");

let cy;
const CJK_FONT_STACK =
  '"Segoe UI Symbol", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", sans-serif';

nodeHtmlLabel(cytoscape);

const RELATION_COLORS = {
  kTraditionalVariant: "#b65e16",
  kSimplifiedVariant: "#23784c",
  kSemanticVariant: "#255f9c",
  kSpecializedSemanticVariant: "#6a4bc3",
  kSpoofingVariant: "#b23b52",
  kZVariant: "#8b6a2b",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function detailRow(label, value) {
  return `
    <div class="detail-row">
      <span class="detail-label">${escapeHtml(label)}</span>
      <span class="detail-value">${escapeHtml(value ?? "(none)")}</span>
    </div>
  `;
}

function renderSummary(result) {
  if (!result.basic_info) {
    summaryPanel.innerHTML = `<p class="muted">No Unihan entry loaded.</p>`;
    return;
  }

  const info = result.basic_info;
  summaryPanel.innerHTML = `
    <div class="hero-character">
      <div class="hero-glyph">${escapeHtml(result.character)}</div>
      <div>
        <div class="hero-codepoint">${escapeHtml(result.codepoint)}</div>
        <div class="hero-status">${escapeHtml(result.status)}</div>
      </div>
    </div>
    ${detailRow("Definition", info.definition)}
    ${detailRow("Mandarin", info.mandarin)}
    ${detailRow("Cantonese", info.cantonese)}
    ${detailRow("Japanese", info.japanese)}
    ${detailRow("Japanese On", info.japanese_on)}
    ${detailRow("Japanese Kun", info.japanese_kun)}
    ${detailRow("Korean", info.korean)}
    ${detailRow("Total strokes", info.total_strokes)}
    ${detailRow("Radical / Stroke", info.radical_stroke)}
    ${detailRow("Unihan Core", info.unihan_core)}
    <div class="section-divider"></div>
    ${detailRow("KangXi", result.dictionary_references?.kangxi ?? "(none)")}
    ${detailRow("IRG KangXi", result.dictionary_references?.irg_kangxi ?? "(none)")}
    ${detailRow("HanYu", result.dictionary_references?.hanyu ?? "(none)")}
  `;
}

function renderRelations(result) {
  if (!result.variant_relations?.length) {
    relationsPanel.innerHTML = `<p class="muted">No variant information.</p>`;
    return;
  }

  const cards = result.variant_relations
    .map((relation) => {
      const linked = relation.linked?.length
        ? relation.linked
            .map(
              (item) => `
                <div class="linked-chip">
                  <span class="chip-glyph">${escapeHtml(item.character)}</span>
                  <span>${escapeHtml(item.codepoint)}</span>
                </div>
              `
            )
            .join("")
        : `<span class="muted">(none)</span>`;

      return `
        <article class="relation-card">
          <div class="relation-header">
            <span class="relation-pill" style="--pill-color:${RELATION_COLORS[relation.field] ?? "#60746a"}">
              ${escapeHtml(relation.field)}
            </span>
          </div>
          <div class="linked-list">${linked}</div>
        </article>
      `;
    })
    .join("");

  relationsPanel.innerHTML = cards;
}

function buildNodePositions(nodes, centerX, centerY, radius) {
  if (nodes.length === 1) {
    return { [nodes[0].codepoint]: { x: centerX, y: centerY } };
  }

  const positions = {};
  const [first, ...rest] = nodes;
  positions[first.codepoint] = { x: centerX, y: centerY };

  rest.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / rest.length - Math.PI / 2;
    positions[node.codepoint] = {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };
  });

  return positions;
}

function destroyGraph() {
  if (cy) {
    cy.destroy();
    cy = null;
  }
}

function relationColor(relation) {
  return RELATION_COLORS[relation] ?? "#6d7f76";
}

function buildGraphElements(result) {
  const nodeElements = (result.component_nodes ?? []).map((node) => ({
    data: {
      id: node.codepoint,
      label: node.character,
      codepoint: node.codepoint,
      definition: node.definition,
      mandarin: node.mandarin,
      isFocus: node.codepoint === result.codepoint ? "true" : "false",
    },
  }));

  const groupedEdges = new Map();
  for (const edge of result.discovered_edges ?? []) {
    const [left, right] =
      edge.source_codepoint <= edge.target_codepoint
        ? [edge.source_codepoint, edge.target_codepoint]
        : [edge.target_codepoint, edge.source_codepoint];
    const pairKey = `${left}|${right}`;
    const relationKey = `${pairKey}|${edge.relation}`;
    const direction =
      edge.source_codepoint <= edge.target_codepoint ? "forward" : "backward";

    if (!groupedEdges.has(relationKey)) {
      groupedEdges.set(relationKey, {
        id: relationKey,
        source: left,
        target: right,
        relation: edge.relation,
        color: relationColor(edge.relation),
        directions: new Set(),
      });
    }

    groupedEdges.get(relationKey).directions.add(direction);
  }

  const groupedList = [...groupedEdges.values()];
  const pairCounts = new Map();
  for (const edge of groupedList) {
    const pairKey = `${edge.source}|${edge.target}`;
    pairCounts.set(pairKey, (pairCounts.get(pairKey) ?? 0) + 1);
  }

  const pairSeen = new Map();
  const edgeElements = groupedList.map((edge) => {
    const pairKey = `${edge.source}|${edge.target}`;
    const pairCount = pairCounts.get(pairKey) ?? 1;
    const pairIndex = pairSeen.get(pairKey) ?? 0;
    pairSeen.set(pairKey, pairIndex + 1);
    const centeredIndex = pairIndex - (pairCount - 1) / 2;
    const curveDistance = pairCount > 1 ? centeredIndex * 34 : 0;
    const bidirectional =
      edge.directions.has("forward") && edge.directions.has("backward");

    return {
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        relation: edge.relation,
        label: edge.relation.replace(/^k/, ""),
        color: edge.color,
        pairCount,
        pairIndex,
        curveDistance,
        bidirectional: bidirectional ? "true" : "false",
      },
    };
  });

  return [...nodeElements, ...edgeElements];
}

function renderGraph(result) {
  const nodes = result.component_nodes ?? [];
  destroyGraph();

  if (!nodes.length) {
    graphPanel.innerHTML = `<div class="graph-empty">No variant graph available for this character.</div>`;
    return;
  }

  graphPanel.innerHTML = `
    <div id="cytoscape-container" class="graph-surface" aria-label="Variant graph"></div>
    <div class="graph-legend">
      ${Object.entries(RELATION_COLORS)
        .map(
          ([field, color]) => `
            <span class="legend-item">
              <span class="legend-swatch" style="background:${color}"></span>
              ${escapeHtml(field)}
            </span>
          `
        )
        .join("")}
    </div>
  `;

  const container = document.querySelector("#cytoscape-container");
  cy = cytoscape({
    container,
    elements: buildGraphElements(result),
    layout: {
      name: "cose",
      animate: false,
      fit: true,
      padding: 36,
      idealEdgeLength: 180,
      nodeRepulsion: 180000,
    },
    style: [
      {
        selector: "node",
        style: {
          width: 72,
          height: 72,
          "background-color": "#f7fbf5",
          "border-width": 3,
          "border-color": "#5a7267",
          label: "data(codepoint)",
          color: "#203028",
          "font-size": 12,
          "font-family": "Cascadia Code, Consolas, monospace",
          "text-valign": "bottom",
          "text-margin-y": 10,
          "overlay-opacity": 0,
        },
      },
      {
        selector: 'node[isFocus = "true"]',
        style: {
          width: 92,
          height: 92,
          "background-color": "#335c4a",
          "border-color": "#264839",
          color: "#ffffff",
        },
      },
      {
        selector: "edge",
        style: {
          width: 4,
          "line-color": "data(color)",
          "curve-style": "straight",
          "source-arrow-shape": "none",
          "source-arrow-color": "data(color)",
          "target-arrow-shape": "triangle",
          "target-arrow-color": "data(color)",
          label: "data(label)",
          "font-size": 12,
          "font-family": CJK_FONT_STACK,
          color: "#42534b",
          "text-background-color": "rgba(255,255,255,0.94)",
          "text-background-opacity": 1,
          "text-background-padding": 4,
          "text-border-color": "rgba(51,92,74,0.12)",
          "text-border-opacity": 1,
          "text-border-width": 1,
          "text-rotation": "autorotate",
          "text-margin-y": -8,
          "overlay-opacity": 0,
        },
      },
      {
        selector: 'edge[bidirectional = "true"]',
        style: {
          "source-arrow-shape": "triangle",
        },
      },
      {
        selector: 'edge[pairCount > 1]',
        style: {
          "curve-style": "unbundled-bezier",
          "control-point-distances": "data(curveDistance)",
          "control-point-weights": 0.5,
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-color": "#d9a441",
          "border-width": 5,
        },
      },
    ],
  });

  cy.nodeHtmlLabel([
    {
      query: "node",
      halign: "center",
      valign: "center",
      halignBox: "center",
      valignBox: "center",
      tpl(data) {
        const isFocus = data.isFocus === "true";
        return `
          <div class="node-html-label ${isFocus ? "focus-node-label" : ""}">
            ${escapeHtml(data.label)}
          </div>
        `;
      },
    },
  ]);

  cy.on("tap", "node", (event) => {
    const node = event.target.data();
    if (node?.label) {
      input.value = node.label;
      runLookup();
    }
  });
}

function renderResult(result) {
  rawJson.textContent = JSON.stringify(result, null, 2);
  renderSummary(result);
  renderRelations(result);
  renderGraph(result);
}

async function runLookup() {
  const character = (input.value || "").trim();
  if (!character) {
    rawJson.textContent = "Please enter a character.";
    return;
  }

  button.disabled = true;
  button.textContent = "Loading...";

  try {
    const result = await invoke("lookup_character", { character });
    renderResult(result);
  } catch (error) {
    const message = `Backend error:\n${String(error)}`;
    rawJson.textContent = message;
    summaryPanel.innerHTML = `<p class="muted">${escapeHtml(message)}</p>`;
    relationsPanel.innerHTML = `<p class="muted">No relation data.</p>`;
    graphPanel.innerHTML = `<div class="graph-empty">Graph render failed.</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Lookup";
  }
}

button.addEventListener("click", runLookup);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runLookup();
  }
});

runLookup();

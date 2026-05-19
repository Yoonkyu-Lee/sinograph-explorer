// Variant relationship graph — the cytoscape view of one character's
// variant family. Owns the cytoscape instance and the relation-label map.
import cytoscape from "cytoscape";
import cola from "cytoscape-cola";
import { invoke } from "@tauri-apps/api/core";
import { CJK_FONT, setStatus } from "./util.js";

cytoscape.use(cola);

let cy = null; // the current cytoscape instance, if the graph is open

// node-tap navigation callback — main.js injects `openEntry` via setNavigate
let navigate = () => {};
export function setNavigate(fn) {
  navigate = fn;
}

// edge relation -> line color (simplified/traditional, semantic, else green)
const EDGE_BUCKETS = [
  { test: /^(simplified|traditional)$/, color: "#b65e16" },
  {
    test: /^(semantic|specialized_semantic|z_variants|spoofing|kanjidic2_resolved)$/,
    color: "#255f9c",
  },
];

function edgeColor(relation) {
  for (const b of EDGE_BUCKETS) if (b.test.test(relation)) return b.color;
  return "#2a7d2e";
}

// variant relation key -> Korean label
const RELATION_KO = {
  traditional: "번체",
  simplified: "간체",
  z_variants: "이체(Z)",
  spoofing: "혼동자",
  semantic: "통용자",
  specialized_semantic: "부분통용",
  kanjidic2_resolved: "일본이체",
  ehanja_dongja: "동자",
  ehanja_bonja: "본자",
  ehanja_sokja: "속자",
  ehanja_yakja: "약자",
  ehanja_goja: "고자",
  ehanja_waja: "와자",
  ehanja_tongja: "통자",
  ehanja_simple: "간체",
  ehanja_hDup: "중복자",
  ehanja_kanji: "일본자",
  ehanja_synonyms: "유의자",
  ehanja_opposites: "반의자",
  ehanja_alt_forms: "이표기",
};

// Korean label for a relation key (used by the graph and the variants panel).
export function relationKo(relation) {
  return RELATION_KO[relation] || relation.replace(/^ehanja_/, "");
}

// Tear down the cytoscape instance, if any.
export function destroyGraph() {
  if (cy) {
    cy.destroy();
    cy = null;
  }
}

// Toggle the variant graph for `codepoint` open/closed.
export async function toggleGraph(codepoint) {
  const section = document.querySelector("#graph-section");
  const toggle = document.querySelector("#graph-toggle");
  if (!section) return;
  if (cy) {
    destroyGraph();
    section.classList.add("hidden");
    if (toggle) toggle.textContent = "관계도 그래프 ▾";
    return;
  }
  try {
    const g = await invoke("variant_graph", { query: codepoint });
    const elements = [
      ...g.nodes.map((n) => ({
        data: { id: n.codepoint, label: n.character, focus: n.focus ? "y" : "n" },
      })),
      ...g.edges.map((ed) => ({
        data: {
          id: `${ed.source}|${ed.target}|${ed.relation}`,
          source: ed.source,
          target: ed.target,
          label: relationKo(ed.relation),
          color: edgeColor(ed.relation),
        },
      })),
    ];
    section.classList.remove("hidden");
    cy = cytoscape({
      container: document.querySelector("#cy"),
      elements,
      layout: {
        // cola — a continuous physics simulation. With `infinite: true` it
        // never stops, so dragging a node makes the connected nodes follow
        // and the whole graph keeps settling, like a living structure.
        // `avoidOverlap` + `nodeSpacing` guarantee a minimum node distance.
        name: "cola",
        animate: true,
        infinite: true,
        fit: false,
        randomize: true,
        avoidOverlap: true,
        nodeSpacing: 14,
        edgeLength: 130,
        handleDisconnected: true,
      },
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#e7efe8",
            "border-color": "#5a7267",
            "border-width": 2,
            label: "data(label)",
            "font-family": CJK_FONT,
            "font-size": 22,
            "text-valign": "center",
            "text-halign": "center",
            color: "#23281f",
            width: 46,
            height: 46,
          },
        },
        {
          selector: 'node[focus = "y"]',
          style: {
            "background-color": "#335c4a",
            "border-color": "#284838",
            color: "#ffffff",
            width: 58,
            height: 58,
            "font-size": 28,
          },
        },
        {
          selector: "edge",
          style: {
            "line-color": "data(color)",
            width: 2.5,
            "curve-style": "bezier",
            label: "data(label)",
            "font-family": CJK_FONT,
            "font-size": 11,
            // label color follows the edge color; no background (transparent)
            color: "data(color)",
            "text-background-opacity": 0,
            // run the label parallel to the edge and lift it off the line
            "text-rotation": "autorotate",
            "text-margin-y": -7,
          },
        },
      ],
    });
    cy.on("tap", "node", (ev) => navigate(ev.target.id()));
    if (toggle) toggle.textContent = "관계도 그래프 ▴";
    section.scrollIntoView({ behavior: "smooth", block: "nearest" });
    // frame the graph once after the initial settle (infinite layout never
    // emits layoutstop, so fit manually instead of fit: true)
    setTimeout(() => {
      if (cy) cy.fit(undefined, 50);
    }, 700);
  } catch (err) {
    setStatus(String(err), true);
  }
}

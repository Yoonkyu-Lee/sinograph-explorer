// Character detail view — the hero glyph plus five panels (구조 / 발음 / 뜻 /
// 이체자 / 급수). Each `renderX` builds an HTML string from the entry data;
// `renderEntry` writes them into the page and wires the graph toggle.
import {
  charToCp,
  escapeHtml,
  hanjaLink,
  hunumPairHtml,
  panel,
  READING_LABELS,
  showView,
} from "./util.js";
import { destroyGraph, relationKo, toggleGraph } from "./graph.js";

const entryEl = document.querySelector("#entry");

// 구조 panel — IDS decomposition, radical, stroke counts.
function renderStructure(e) {
  const s = e.structure;
  const rows = [];
  if (s.primary_ids) {
    const components = new Set(e.ids_components.map((c) => c.codepoint));
    const ids = [...s.primary_ids]
      .map((ch) => {
        const cp = charToCp(ch);
        return components.has(cp)
          ? hanjaLink(cp, ch)
          : `<span class="idc">${escapeHtml(ch)}</span>`;
      })
      .join("");
    const idc = s.ids_top_idc
      ? ` <span class="muted">(${escapeHtml(s.ids_top_idc)})</span>`
      : "";
    rows.push(`<div class="kv"><span class="k">분해</span>
      <span class="v ids-line">${ids}${idc}</span></div>`);
  }
  if (s.radical_idx != null) {
    let rad = String(s.radical_idx);
    if (s.radical_char) {
      rad += " " + hanjaLink(charToCp(s.radical_char), s.radical_char, "small");
    }
    if (s.radical_name) rad += ` <span class="muted">${escapeHtml(s.radical_name)}</span>`;
    if (s.radical_strokes != null) rad += ` <span class="muted">· ${s.radical_strokes}획</span>`;
    rows.push(`<div class="kv"><span class="k">부수</span><span class="v">${rad}</span></div>`);
  }
  const strokes = [];
  if (s.total_strokes != null) strokes.push(`총 ${s.total_strokes}획`);
  if (s.residual_strokes != null) strokes.push(`잔여 ${s.residual_strokes}획`);
  if (strokes.length) {
    rows.push(`<div class="kv"><span class="k">획수</span>
      <span class="v">${escapeHtml(strokes.join(" · "))}</span></div>`);
  }
  return rows.length ? rows.join("") : `<p class="muted">구조 정보 없음</p>`;
}

// 발음 panel — Korean 훈음 first, then the five non-Korean readings.
function renderReadings(e) {
  const rows = [];
  if (e.hunum && e.hunum.length) {
    const pairs = e.hunum
      .map(hunumPairHtml)
      .join(`<span class="pair-sep"> · </span>`);
    rows.push(`<div class="kv"><span class="k">한국</span>
      <span class="v hunum-v">${pairs}</span></div>`);
  }
  for (const [key, label] of READING_LABELS) {
    const vals = e.readings[key] || [];
    if (vals.length) {
      rows.push(`<div class="kv"><span class="k">${escapeHtml(label)}</span>
        <span class="v">${escapeHtml(vals.join(" / "))}</span></div>`);
    }
  }
  return rows.length ? rows.join("") : `<p class="muted">발음 정보 없음</p>`;
}

// 뜻 panel — Korean and English meanings.
function renderMeanings(e) {
  const rows = [];
  if (e.meanings.ko && e.meanings.ko.length) {
    rows.push(`<div class="kv"><span class="k">한국어</span>
      <span class="v">${escapeHtml(e.meanings.ko.join(" / "))}</span></div>`);
  }
  if (e.meanings.en && e.meanings.en.length) {
    rows.push(`<div class="kv"><span class="k">영어</span>
      <span class="v">${escapeHtml(e.meanings.en.join("; "))}</span></div>`);
  }
  return rows.length ? rows.join("") : `<p class="muted">뜻 정보 없음</p>`;
}

// 이체자 panel — family chips, variant/related edges, graph toggle button.
function renderVariants(e) {
  const blocks = [];
  if (e.family && e.family.size > 1) {
    const chips = e.family.members
      .map((m) =>
        hanjaLink(m.codepoint, m.character, m.codepoint === e.codepoint ? "self" : "")
      )
      .join("");
    blocks.push(`<div class="kv"><span class="k">family (${e.family.size})</span>
      <span class="v chip-row">${chips}</span></div>`);
  }
  const edgeLine = (edges) =>
    edges
      .map(
        (v) => `<span class="edge">${hanjaLink(v.target_codepoint, v.target_character)}
        <span class="rel">${escapeHtml(relationKo(v.relation))}</span></span>`
      )
      .join("");
  const variantEdges = e.variants.filter((v) => v.category === "variant");
  const semanticEdges = e.variants.filter((v) => v.category === "semantic");
  if (variantEdges.length) {
    blocks.push(`<div class="kv"><span class="k">이체 관계</span>
      <span class="v chip-row">${edgeLine(variantEdges)}</span></div>`);
  }
  if (semanticEdges.length) {
    blocks.push(`<div class="kv"><span class="k">관련어</span>
      <span class="v chip-row">${edgeLine(semanticEdges)}</span></div>`);
  }
  if (e.family && e.family.size > 1) {
    blocks.push(`<div class="graph-btn-wrap">
      <button id="graph-toggle" class="graph-btn">관계도 그래프 ▾</button></div>`);
  }
  return blocks.length ? blocks.join("") : `<p class="muted">이체자 정보 없음</p>`;
}

// 급수 panel — grade / level across the four standard sets.
function renderGrades(e) {
  const g = e.grades;
  if (!g) return `<p class="muted">급수 정보 없음</p>`;
  const rows = [];
  const add = (k, v) =>
    rows.push(`<div class="kv"><span class="k">${escapeHtml(k)}</span>
      <span class="v">${escapeHtml(v)}</span></div>`);
  if (g.kr_grade) add("한자검정", g.kr_grade);
  if (g.kr_education) add("한문 교육용", g.kr_education);
  if (g.cn_tonggyong != null) add("중국 통용규범", g.cn_tonggyong + "급");
  if (g.jp_grade != null) {
    let label;
    if (g.jp_grade <= 6) label = g.jp_grade + "학년 (교육한자)";
    else if (g.jp_grade >= 9) label = "인명용한자";
    else label = "상용한자 (중등)";
    add("일본 학년", label);
  }
  if (g.jp_freq != null) add("일본 빈도", "신문 " + g.jp_freq + "위");
  if (g.jp_jlpt != null) add("JLPT", "구 " + g.jp_jlpt + "급");
  if (g.unihan_core) add("Unihan core", "포함 · " + g.unihan_core);
  return rows.length ? rows.join("") : `<p class="muted">급수 정보 없음</p>`;
}

// Render the full entry detail view into the page and show it.
export function renderEntry(e) {
  destroyGraph();
  entryEl.innerHTML = `
    <div class="hero">
      <div class="hero-glyph">${escapeHtml(e.character)}</div>
      <div class="hero-meta">
        <div class="hero-cp">${escapeHtml(e.codepoint)}</div>
        ${e.block ? `<div class="hero-block">${escapeHtml(e.block)}</div>` : ""}
      </div>
    </div>
    <div class="panel-grid">
      ${panel("구조", renderStructure(e))}
      ${panel("발음", renderReadings(e))}
      ${panel("뜻", renderMeanings(e))}
      ${panel("이체자", renderVariants(e))}
      ${panel("급수", renderGrades(e))}
    </div>
    <section id="graph-section" class="graph-section hidden">
      <div class="graph-head">이체자 관계도
        <span class="muted">— 노드 클릭 시 그 글자로 이동</span></div>
      <div id="cy" class="cy"></div>
    </section>
  `;
  showView("entry");
  const toggle = document.querySelector("#graph-toggle");
  if (toggle) toggle.addEventListener("click", () => toggleGraph(e.codepoint));
}

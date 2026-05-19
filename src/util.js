// Shared constants and stateless helper functions used across the frontend.
// This module imports nothing — it is the bottom of the dependency graph.

// font stack for CJK glyphs (used in HTML and in the cytoscape graph)
export const CJK_FONT =
  '"Noto Serif CJK KR", "Malgun Gothic", "Microsoft JhengHei", serif';

// reading-type key -> Korean label, in display order
export const READING_LABELS = [
  ["mandarin", "표준중국어"],
  ["cantonese", "광동어"],
  ["onyomi", "일본 음독"],
  ["kunyomi", "일본 훈독"],
  ["vietnamese", "베트남어"],
];

const RECENT_KEY = "sino.recent";

// static DOM elements (this module runs after the document is parsed)
const statusEl = document.querySelector("#status");
const homeEl = document.querySelector("#home");
const resultsEl = document.querySelector("#results");
const entryEl = document.querySelector("#entry");

// Escape a value for safe insertion into HTML.
export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// One character -> its "U+XXXX" codepoint string.
export function charToCp(ch) {
  return "U+" + ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0");
}

// Should this query be a direct lookup (a single hanzi or a U+XXXX code)?
export function isLookupQuery(q) {
  q = q.trim();
  if (/^u\+?[0-9a-fA-F]{4,}$/.test(q)) return true;
  const chars = [...q];
  return chars.length === 1 && /\p{Script=Han}/u.test(chars[0]);
}

// A clickable hanzi button (navigation is handled by event delegation).
export function hanjaLink(cp, ch, extraClass = "") {
  const glyph = ch && ch.length ? ch : "?";
  return `<button class="hanja-link ${extraClass}" data-cp="${escapeHtml(cp)}"
    >${escapeHtml(glyph)}</button>`;
}

// Show a status / error message under the search bar.
export function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

// Show exactly one of the three top-level views: home / results / entry.
export function showView(name) {
  homeEl.classList.toggle("hidden", name !== "home");
  resultsEl.classList.toggle("hidden", name !== "results");
  entryEl.classList.toggle("hidden", name !== "entry");
}

// Read the "recently viewed" list from localStorage.
export function getRecent() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY)) || [];
  } catch {
    return [];
  }
}

// Add a character to the front of the "recently viewed" list (max 14).
export function pushRecent(cp, ch) {
  const recent = getRecent().filter((x) => x.cp !== cp);
  recent.unshift({ cp, ch });
  localStorage.setItem(RECENT_KEY, JSON.stringify(recent.slice(0, 14)));
}

// One titled panel section (used to lay out the entry detail view).
export function panel(title, bodyHtml) {
  return `<section class="panel">
    <h2>${escapeHtml(title)}</h2>
    <div class="panel-body">${bodyHtml}</div>
  </section>`;
}

// A small (훈) / (음) label, set like a lower-right subscript.
export function hunumTag(label) {
  return `<sub class="rk-tag">(${escapeHtml(label)})</sub>`;
}

// One 훈음 pair — 자훈(훈) 독음(음); 자훈 is omitted when absent.
export function hunumPairHtml(h) {
  const jahun = h.jahun
    ? `<span class="jahun">${escapeHtml(h.jahun)}${hunumTag("훈")}</span> `
    : "";
  return `${jahun}<span class="dokeum">${escapeHtml(h.dokeum)}${hunumTag("음")}</span>`;
}

// Navigation — the back-stack state and the actions that move between views
// (open an entry, run a search, browse a radical, go home, go back). This
// module owns the navigation state; rendering is delegated to home.js and
// entry.js, and clicks are wired by main.js via event delegation.
import { invoke } from "@tauri-apps/api/core";
import {
  escapeHtml,
  isLookupQuery,
  pushRecent,
  setStatus,
  showView,
} from "./util.js";
import { renderHome } from "./home.js";
import { renderEntry } from "./entry.js";

const crumbEl = document.querySelector("#crumb");
const resultsEl = document.querySelector("#results");
const queryInput = document.querySelector("#query");

let backStack = []; // codepoints visited, for the back button
let currentCp = null; // codepoint of the entry currently shown
let homeData = null; // cached home payload for the session

// Render the breadcrumb trail. Buttons carry IDs only — main.js handles
// their clicks through event delegation.
function renderCrumb() {
  if (!backStack.length) {
    crumbEl.classList.add("hidden");
    crumbEl.innerHTML = "";
    return;
  }
  crumbEl.classList.remove("hidden");
  const trail = backStack
    .slice(-8)
    .map((cp) => escapeHtml(cp))
    .join("  ›  ");
  crumbEl.innerHTML = `<button id="back-btn">← 뒤로</button>
    <button id="home-btn">홈</button>
    <span class="crumb-trail">${trail}</span>`;
}

// Render a list of search / radical-browse hits into the results view.
function renderResults(label, hits) {
  if (!hits.length) {
    resultsEl.innerHTML = `<p class="muted">${escapeHtml(label)} — 결과 없음.</p>`;
    showView("results");
    return;
  }
  const items = hits
    .map(
      (h) => `<li class="result-item" data-cp="${escapeHtml(h.codepoint)}">
        <span class="result-glyph">${escapeHtml(h.character)}</span>
        <span class="result-cp">${escapeHtml(h.codepoint)}</span>
        <span class="result-gloss">${escapeHtml(h.gloss)}</span>
      </li>`
    )
    .join("");
  resultsEl.innerHTML = `<p class="results-head">${escapeHtml(label)}</p>
    <ul class="result-list">${items}</ul>`;
  showView("results");
}

// Open one character's full dictionary entry. `fromNav` is set when the back
// button drives the navigation, so the codepoint is not re-pushed.
export async function openEntry(query, { fromNav = false } = {}) {
  try {
    setStatus("조회 중…");
    const entry = await invoke("lookup", { query });
    if (currentCp && !fromNav && currentCp !== entry.codepoint) {
      backStack.push(currentCp);
    }
    currentCp = entry.codepoint;
    renderEntry(entry);
    renderCrumb();
    pushRecent(entry.codepoint, entry.character);
    setStatus("");
  } catch (err) {
    setStatus(String(err), true);
  }
}

// Run a reverse (meaning / reading) search and show the results.
export async function runSearch(query) {
  try {
    setStatus("검색 중…");
    const hits = await invoke("search", { query, limit: 80 });
    renderResults(`"${query}" 검색 — ${hits.length}건`, hits);
    renderCrumb();
    setStatus("");
  } catch (err) {
    setStatus(String(err), true);
  }
}

// List the characters under one Kangxi radical.
export async function openRadical(idx, char, name) {
  try {
    setStatus("부수 글자 조회 중…");
    const hits = await invoke("radical_chars", { radicalIdx: idx, limit: 400 });
    renderResults(`부수 ${idx} ${char} ${name} — ${hits.length}자`, hits);
    renderCrumb();
    setStatus(hits.length >= 400 ? "획수 적은 순 400자까지 표시." : "");
  } catch (err) {
    setStatus(String(err), true);
  }
}

// Handle the search bar: a single hanzi / U+XXXX is a lookup, else a search.
export function submitQuery() {
  const q = queryInput.value.trim();
  if (!q) return;
  if (isLookupQuery(q)) openEntry(q);
  else runSearch(q);
}

// Go back to the previously viewed entry.
export function goBack() {
  const prev = backStack.pop();
  if (prev) openEntry(prev, { fromNav: true });
}

// Load and show the home screen (home_data is fetched once and cached).
export async function loadHome() {
  try {
    setStatus("");
    if (!homeData) homeData = await invoke("home_data");
    renderHome(homeData);
    showView("home");
    currentCp = null;
    backStack = [];
    crumbEl.classList.add("hidden");
    queryInput.value = "";
  } catch (err) {
    setStatus(String(err), true);
  }
}

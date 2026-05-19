// Capture overlay — cursor-following 유효범위 rectangle + live recognition.
//
// Flow: the rectangle tracks the cursor and recognition runs on every pause
// (live preview). A click *locks* the rectangle in place so the candidate
// panel holds still and can be clicked. Clicking a candidate opens it in the
// dictionary; clicking empty space again re-arms tracking; Esc cancels.
import "./overlay.css";
import { invoke } from "@tauri-apps/api/core";

const MIN_SIZE = 48;
const MAX_SIZE = 360;
const DEFAULT_SIZE = 120;

const root = document.querySelector("#overlay-root");

// rectangle size in CSS (logical) pixels; Rust scales it to physical px
let rectSize = DEFAULT_SIZE;
let rectX = window.innerWidth / 2;
let rectY = window.innerHeight / 2;
let frozen = false;

// ---- elements ----
const rect = document.createElement("div");
rect.id = "capture-rect";
root.appendChild(rect);

const panel = document.createElement("div");
panel.id = "cand-panel";
panel.className = "hidden";
root.appendChild(panel);

const hint = document.createElement("div");
hint.className = "hint";
root.appendChild(hint);

// ---- layout ----
function layout() {
  const half = rectSize / 2;
  rect.style.width = `${rectSize}px`;
  rect.style.height = `${rectSize}px`;
  rect.style.left = `${rectX - half}px`;
  rect.style.top = `${rectY - half}px`;
  positionPanel();
}

function positionPanel() {
  const gap = 14;
  const half = rectSize / 2;
  const pw = panel.offsetWidth || 150;
  let left = rectX + half + gap;
  if (left + pw > window.innerWidth) left = rectX - half - gap - pw;
  const ph = panel.offsetHeight || 120;
  let top = rectY - half;
  if (top + ph > window.innerHeight) top = window.innerHeight - ph - 8;
  panel.style.left = `${Math.max(8, left)}px`;
  panel.style.top = `${Math.max(8, top)}px`;
}

function updateChrome() {
  rect.classList.toggle("locked", frozen);
  hint.innerHTML = frozen
    ? "후보 한자를 클릭 · 빈 곳 클릭 시 다시 조준 · <b>Esc</b> 취소"
    : "한자 위에서 <b>클릭하여 고정</b> · 휠로 크기 조절 · <b>Esc</b> 취소";
}

// ---- candidate panel ----
function renderCandidates(cands) {
  if (!cands || cands.length === 0) {
    panel.className = "hidden";
    return;
  }
  const rows = cands
    .map(
      (c) => `<div class="cand" data-cp="${c.codepoint}">
        <span class="cand-glyph">${c.character}</span>
        <span class="cand-cp">${c.codepoint}</span>
        <span class="cand-score">${Math.round(c.score * 100)}%</span>
      </div>`
    )
    .join("");
  const head = frozen
    ? "후보 한자 — 클릭하여 검색"
    : "후보 (클릭하여 고정)";
  panel.innerHTML = `<div class="cand-head">${head}</div>${rows}`;
  panel.className = "";
  positionPanel();
}

// ---- live recognition loop ----
// One inference at a time; the trailing rerun guarantees the last pause
// always gets a fresh result.
const DEBOUNCE_MS = 300;
let debounceTimer = null;
let inferInFlight = false;
let pendingRerun = false;

function scheduleRecognize() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runRecognize, DEBOUNCE_MS);
}

async function runRecognize() {
  if (inferInFlight) {
    pendingRerun = true;
    return;
  }
  inferInFlight = true;
  try {
    const cands = await invoke("recognize_under_cursor", {
      size: rectSize,
      k: 6,
    });
    renderCandidates(cands);
  } catch (_) {
    // transient capture/inference errors are ignored — next pause retries
  } finally {
    inferInFlight = false;
    if (pendingRerun) {
      pendingRerun = false;
      runRecognize();
    }
  }
}

// ---- input ----
window.addEventListener("mousemove", (ev) => {
  if (frozen) return; // locked — let the cursor reach the panel
  rectX = ev.clientX;
  rectY = ev.clientY;
  layout();
  scheduleRecognize();
});

window.addEventListener("click", (ev) => {
  const row = ev.target.closest(".cand");
  if (row && row.dataset.cp) {
    invoke("focus_main_with_lookup", { codepoint: row.dataset.cp }).catch(
      () => {}
    );
    return;
  }
  // clicked the overlay (not a candidate) — toggle the lock
  frozen = !frozen;
  if (frozen) {
    rectX = ev.clientX;
    rectY = ev.clientY;
    layout();
    runRecognize(); // refresh candidates for the locked spot now
  } else {
    renderCandidates([]);
  }
  updateChrome();
});

window.addEventListener(
  "wheel",
  (ev) => {
    ev.preventDefault();
    const step = ev.deltaY > 0 ? -16 : 16;
    rectSize = Math.min(MAX_SIZE, Math.max(MIN_SIZE, rectSize + step));
    layout();
    scheduleRecognize();
  },
  { passive: false }
);

window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    invoke("close_capture_overlay").catch(() => {});
  }
});

updateChrome();
layout();

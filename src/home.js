// Home screen rendering — the daily-character card, recently-viewed chips,
// database stats, and the 214-radical browse grid. `renderHome` is a pure
// renderer: it takes the `home_data` payload and fills the #home section.
import { escapeHtml, getRecent, hanjaLink, hunumPairHtml } from "./util.js";

const homeEl = document.querySelector("#home");

// Render the home dashboard from a `home_data` payload.
export function renderHome(data) {
  const d = data.daily;
  const dailyHtml = d
    ? `<button class="daily-card nav-card" data-cp="${escapeHtml(d.codepoint)}">
        <div class="daily-glyph">${escapeHtml(d.character)}</div>
        <div class="daily-info">
          <div class="daily-label">오늘의 한자</div>
          ${d.hunum ? `<div class="daily-hunum">${hunumPairHtml(d.hunum)}</div>` : ""}
          <div class="daily-gloss">${escapeHtml(d.gloss || "(뜻 정보 없음)")}</div>
          <div class="daily-meta">${escapeHtml(d.codepoint)}${
        d.primary_ids ? " · " + escapeHtml(d.primary_ids) : ""
      }${d.total_strokes != null ? " · " + d.total_strokes + "획" : ""}</div>
        </div>
      </button>`
    : "";

  const recent = getRecent();
  const recentHtml = recent.length
    ? recent.map((x) => hanjaLink(x.cp, x.ch)).join("")
    : `<span class="muted">아직 본 글자가 없습니다.</span>`;

  const radicalCells = data.radicals
    .map((r) => {
      const ch = r.character || "?";
      const name = (r.name_ko || "").replace(/部$/, "");
      return `<button class="radical-cell" data-idx="${r.radical_idx}"
        data-char="${escapeHtml(ch)}" data-name="${escapeHtml(r.name_ko || "")}">
        <span class="rad-char">${escapeHtml(ch)}</span>
        <span class="rad-meta">${r.radical_idx} · ${escapeHtml(name)}</span>
        <span class="rad-count">${r.char_count.toLocaleString()}</span>
      </button>`;
    })
    .join("");

  const s = data.stats;
  homeEl.innerHTML = `
    <div class="home-top">
      ${dailyHtml}
      <div class="home-recent">
        <h2>최근 본 글자</h2>
        <div class="chip-row">${recentHtml}</div>
        <h2 class="stat-head">데이터베이스</h2>
        <ul class="stat-list">
          <li>한자 <b>${s.characters.toLocaleString()}</b>자</li>
          <li>발음 보유 <b>${s.with_reading.toLocaleString()}</b>자</li>
          <li>뜻 보유 <b>${s.with_meaning.toLocaleString()}</b>자</li>
          <li>이체자 family <b>${s.variant_families.toLocaleString()}</b>개</li>
        </ul>
      </div>
    </div>
    <div class="home-radicals">
      <h2>부수로 찾기 · 214</h2>
      <div class="radical-grid">${radicalCells}</div>
    </div>
  `;
}

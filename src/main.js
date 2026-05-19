// Orchestrator — the app's entry point. Wires the static UI (search bar,
// buttons) to the navigation actions, sets up event delegation for
// dynamically rendered controls, and starts the app on the home screen.
import "./styles.css";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { setStatus } from "./util.js";
import { setNavigate } from "./graph.js";
import { goBack, loadHome, openEntry, openRadical, submitQuery } from "./nav.js";

const queryInput = document.querySelector("#query");
const goButton = document.querySelector("#go");
const recognizeButton = document.querySelector("#recognize");
const homeLink = document.querySelector("#home-link");

// --- search bar ---
goButton.addEventListener("click", submitQuery);
queryInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") submitQuery();
});
homeLink.addEventListener("click", loadHome);

// --- on-screen hanzi recognition overlay ---
recognizeButton.addEventListener("click", () => {
  invoke("open_capture_overlay").catch((e) => setStatus(String(e), true));
});

// --- event delegation for dynamically rendered controls ---
// Every clickable hanzi carries a `data-cp` attribute; crumb / radical
// buttons are matched by id / class. This one listener covers them all.
document.addEventListener("click", (ev) => {
  if (ev.target.closest("#back-btn")) {
    goBack();
    return;
  }
  if (ev.target.closest("#home-btn")) {
    loadHome();
    return;
  }
  const rad = ev.target.closest(".radical-cell");
  if (rad) {
    openRadical(Number(rad.dataset.idx), rad.dataset.char, rad.dataset.name);
    return;
  }
  const nav = ev.target.closest(".hanja-link, .result-item, .nav-card");
  if (nav && nav.dataset.cp) openEntry(nav.dataset.cp);
});

// a candidate picked in the capture overlay -> open that entry
listen("lookup-request", (ev) => {
  if (ev.payload) openEntry(String(ev.payload));
});

// let the variant graph navigate to a tapped node
setNavigate(openEntry);

// initial view — the home screen
loadHome();

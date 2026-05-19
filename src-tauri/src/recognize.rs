//! SCER hanzi recognition — ONNX inference (tract) + cosine NN over the
//! anchor DB, plus the two recognition commands. doc/39 M1 / M5.
//!
//! The SCER model maps a 128x128 RGB glyph image to a 128-d L2-normalized
//! embedding. Recognition is a nearest-neighbour search of that embedding
//! against `scer_anchor_db_v20.npy` — one anchor row per class. The class
//! index maps a row back to a `U+XXXX` codepoint.

use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use serde::Serialize;
use tauri::State;
use tract_onnx::prelude::*;

const INPUT_SIZE: usize = 128;

/// A fully optimized, runnable tract plan for the SCER ONNX graph.
type ScerPlan =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// One recognition candidate returned to the frontend.
#[derive(Serialize, Clone)]
pub struct Candidate {
    pub codepoint: String,
    pub character: String,
    /// cosine similarity to the anchor, in [-1, 1]
    pub score: f32,
}

/// The recognition engine — model + anchor table + index map. Held in managed
/// state behind an `Arc`; `recognize` takes `&self` so it is safe to share.
pub struct Recognizer {
    model: ScerPlan,
    anchors: Array2<f32>,   // (n_class, 128), L2-normalized rows
    idx_to_cp: Vec<String>, // class index -> "U+XXXX"
}

impl Recognizer {
    /// Load the ONNX model, anchor DB, and class index from resource files.
    pub fn load(
        onnx_path: &Path,
        anchor_path: &Path,
        class_index_path: &Path,
    ) -> Result<Self, String> {
        let model = onnx()
            .model_for_path(onnx_path)
            .map_err(|e| format!("onnx load: {e}"))?
            .with_input_fact(
                0,
                f32::fact([1, 3, INPUT_SIZE, INPUT_SIZE]).into(),
            )
            .map_err(|e| format!("onnx input fact: {e}"))?
            .into_optimized()
            .map_err(|e| format!("onnx optimize: {e}"))?
            .into_runnable()
            .map_err(|e| format!("onnx runnable: {e}"))?;

        let anchors: Array2<f32> = ndarray_npy::read_npy(anchor_path)
            .map_err(|e| format!("anchor npy: {e}"))?;

        let raw = std::fs::read_to_string(class_index_path)
            .map_err(|e| format!("class index read: {e}"))?;
        let map: std::collections::HashMap<String, usize> =
            serde_json::from_str(&raw).map_err(|e| format!("class index json: {e}"))?;
        let n_class = anchors.shape()[0];
        let mut idx_to_cp = vec![String::new(); n_class];
        for (cp, idx) in map {
            if idx < n_class {
                idx_to_cp[idx] = cp;
            }
        }

        Ok(Self {
            model,
            anchors,
            idx_to_cp,
        })
    }

    /// Recognize one glyph image — returns the top-`k` candidates, best first.
    pub fn recognize(
        &self,
        img: &image::RgbImage,
        k: usize,
    ) -> Result<Vec<Candidate>, String> {
        let input = preprocess(img);
        let tensor =
            Tensor::from_shape(&[1, 3, INPUT_SIZE, INPUT_SIZE], &input)
                .map_err(|e| format!("tensor: {e}"))?;
        let out = self
            .model
            .run(tvec!(tensor.into()))
            .map_err(|e| format!("infer: {e}"))?;
        let view = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("output: {e}"))?;

        // L2-normalize the embedding (model already normalizes, but a fresh
        // normalize keeps the cosine math exact regardless of export quirks).
        let mut emb: Vec<f32> = view.iter().copied().collect();
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in &mut emb {
            *x /= norm;
        }
        let emb = Array1::from(emb);

        // cosine similarity vs every anchor (rows are unit vectors already)
        let sims = self.anchors.dot(&emb); // (n_class,)

        let n = sims.len();
        let k = k.clamp(1, n);
        let mut scored: Vec<(usize, f32)> = sims.iter().copied().enumerate().collect();
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored
            .into_iter()
            .map(|(idx, score)| {
                let codepoint = self.idx_to_cp[idx].clone();
                Candidate {
                    character: cp_to_char(&codepoint),
                    codepoint,
                    score,
                }
            })
            .collect())
    }
}

/// Capture a square screen region centered on a physical-pixel point and
/// return it ready for `Recognizer::recognize`. doc/39 M4 — all crop math is
/// done in physical pixels against the monitor the cursor sits on.
///
/// `size_logical` is the overlay rectangle size in CSS pixels; it is scaled
/// by the target monitor's scale factor to physical pixels.
pub fn capture_square(
    cursor_x: f64,
    cursor_y: f64,
    size_logical: f64,
) -> Result<image::RgbImage, String> {
    let monitors =
        xcap::Monitor::all().map_err(|e| format!("monitors: {e}"))?;
    let (cx, cy) = (cursor_x as i32, cursor_y as i32);
    let mon = monitors
        .into_iter()
        .find(|m| match (m.x(), m.y(), m.width(), m.height()) {
            (Ok(x), Ok(y), Ok(w), Ok(h)) => {
                cx >= x && cx < x + w as i32 && cy >= y && cy < y + h as i32
            }
            _ => false,
        })
        .ok_or("커서가 있는 모니터를 찾지 못했습니다.")?;

    let mx = mon.x().map_err(|e| e.to_string())?;
    let my = mon.y().map_err(|e| e.to_string())?;
    let mw = mon.width().map_err(|e| e.to_string())?;
    let mh = mon.height().map_err(|e| e.to_string())?;
    let scale = mon.scale_factor().map_err(|e| e.to_string())? as f64;

    // logical rectangle size -> physical pixels, clamped to the monitor
    let size = ((size_logical * scale).round() as u32).clamp(16, mw.min(mh));
    // monitor-local physical crop origin, clamped inside the monitor
    let local_cx = cursor_x - mx as f64;
    let local_cy = cursor_y - my as f64;
    let x0 = (local_cx - size as f64 / 2.0)
        .round()
        .clamp(0.0, (mw - size) as f64) as u32;
    let y0 = (local_cy - size as f64 / 2.0)
        .round()
        .clamp(0.0, (mh - size) as f64) as u32;

    let shot = mon.capture_image().map_err(|e| format!("capture: {e}"))?;
    let (sw, sh) = (shot.width(), shot.height());
    // rebuild as our own image::RgbaImage from raw bytes — version-agnostic
    let full = image::RgbaImage::from_raw(sw, sh, shot.into_raw())
        .ok_or("캡처 이미지 변환 실패")?;
    let crop = image::imageops::crop_imm(&full, x0, y0, size, size).to_image();
    Ok(image::DynamicImage::ImageRgba8(crop).to_rgb8())
}

/// `U+XXXX` codepoint string -> the literal character.
fn cp_to_char(cp: &str) -> String {
    cp.strip_prefix("U+")
        .and_then(|h| u32::from_str_radix(h, 16).ok())
        .and_then(char::from_u32)
        .map(|c| c.to_string())
        .unwrap_or_default()
}

/// RGB image -> SCER input tensor data (NCHW, normalized to [-1, 1]).
///
/// Mirrors `deploy_pi/infer_pi_onnx.py:preprocess`: pad to a white square,
/// resize to 128x128 (bilinear), `/255` then `(x-0.5)/0.5`.
fn preprocess(img: &image::RgbImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let side = w.max(h);
    let mut square =
        image::RgbImage::from_pixel(side, side, image::Rgb([255, 255, 255]));
    let ox = ((side - w) / 2) as i64;
    let oy = ((side - h) / 2) as i64;
    image::imageops::overlay(&mut square, img, ox, oy);
    let resized = image::imageops::resize(
        &square,
        INPUT_SIZE as u32,
        INPUT_SIZE as u32,
        image::imageops::FilterType::Triangle, // bilinear
    );

    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = (p[c] as f32 / 255.0 - 0.5) / 0.5;
                out[c * plane + y * INPUT_SIZE + x] = v;
            }
        }
    }
    out
}

// ---- Tauri commands ----
// The `Recognizer` lives in managed state as an `Arc` so it can be cheaply
// cloned into a blocking worker thread. Both commands are `async` and run the
// heavy work (image decode / screen capture + ONNX inference) via
// `spawn_blocking`, keeping the UI thread free.

/// Command: recognize a glyph image file on disk — the M1 headless gate and
/// a debug entry point.
#[tauri::command]
pub async fn recognize_image_file(
    rec: State<'_, Arc<Recognizer>>,
    path: String,
    k: usize,
) -> Result<Vec<Candidate>, String> {
    let rec = rec.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let img = image::open(&path)
            .map_err(|e| format!("image open: {e}"))?
            .to_rgb8();
        rec.recognize(&img, k)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Command: live recognition for the capture overlay — captures the screen
/// square under the cursor and returns the top-`k` candidates. doc/39 M5.
/// Cursor position is read here in Rust (physical px); the overlay passes only
/// the rectangle size in CSS pixels.
#[tauri::command]
pub async fn recognize_under_cursor(
    app: tauri::AppHandle,
    rec: State<'_, Arc<Recognizer>>,
    size: f64,
    k: usize,
) -> Result<Vec<Candidate>, String> {
    let rec = rec.inner().clone();
    let cursor = app.cursor_position().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let crop = capture_square(cursor.x, cursor.y, size)?;
        rec.recognize(&crop, k)
    })
    .await
    .map_err(|e| e.to_string())?
}

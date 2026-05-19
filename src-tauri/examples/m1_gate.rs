//! doc/39 GATE M1 — exercises the exact `recognize::Recognizer` path the
//! `recognize_image_file` command uses, without the Tauri UI.
//!
//!   cargo run --example m1_gate
//!
//! PASS: top-1 >= 60% on deploy_pi/test_chars (filenames are the ground truth).

use std::path::PathBuf;
use std::time::Instant;

use sinograph_explorer_lib::recognize::Recognizer;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let res = manifest.join("resources");
    let test_dir = manifest.join("../../deploy_pi/test_chars");

    println!("[m1] loading recognizer...");
    let t0 = Instant::now();
    let rec = Recognizer::load(
        &res.join("scer_v4.onnx"),
        &res.join("scer_anchor_db_v20.npy"),
        &res.join("class_index.json"),
    )
    .expect("recognizer load");
    println!("[m1] loaded in {:?}", t0.elapsed());

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&test_dir)
        .expect("test_chars dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    paths.sort();

    let (mut top1, mut top5) = (0usize, 0usize);
    let mut total_ms = 0f64;
    println!("[m1] {} test images\n", paths.len());

    for path in &paths {
        let gt: String = path.file_stem().unwrap().to_string_lossy().into_owned();
        let img = image::open(path).expect("open image").to_rgb8();
        let t = Instant::now();
        let cands = rec.recognize(&img, 5).expect("recognize");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;

        let chars: Vec<&str> = cands.iter().map(|c| c.character.as_str()).collect();
        let hit1 = chars.first() == Some(&gt.as_str());
        let hit5 = chars.contains(&gt.as_str());
        top1 += hit1 as usize;
        top5 += hit5 as usize;
        let mark = if hit1 { "OK " } else if hit5 { "o5 " } else { "XX " };
        println!(
            "  {mark} gt={gt}  top5={}  sim={:.3}  {ms:.0}ms",
            chars.join(" "),
            cands[0].score,
        );
    }

    let n = paths.len();
    println!("\n{}", "=".repeat(56));
    println!("[m1] GATE — {n} images");
    println!("  top-1: {top1}/{n}  ({:.1}%)", 100.0 * top1 as f64 / n as f64);
    println!("  top-5: {top5}/{n}  ({:.1}%)", 100.0 * top5 as f64 / n as f64);
    println!("  avg latency: {:.0}ms", total_ms / n as f64);
    if top1 < (n * 6) / 10 {
        eprintln!("[m1] GATE FAIL — top-1 too low");
        std::process::exit(1);
    }
    println!("[m1] GATE M1 PASS");
}

# Lab 3 Proposal (revised) — ECE 479
**Project Title:** Sinograph Explorer
**Track:** Track 2 — Custom DNN Model


---

## Track Selection

This project follows **Track 2 (Custom DNN Model)**. The goal is to recognize rare, variant, and historical CJK characters — including cases where standard OCR libraries fail because they are biased toward high-frequency modern forms and trained in a language-specific mode (a Japanese-oriented model handles shinjitai well but struggles with Korean-specific hanja or CJK Extension B rare forms). This project therefore treats hanja / kanji / hanzi as a shared **visual character system** and builds a single DNN that classifies each input image by its Unicode codepoint identity, independent of language. The original proposal framed the work as an IoT pipeline (Track 1); during midpoint we confirmed that the scientifically novel and measurable contribution lives in **the classifier itself**, not the edge integration. Pi + Coral deployment remains a reference target (the Lab-2 ONNX→TFLite path still applies) but it is not where the technical contribution is scored.

---

## System Description and Novelty

The system is a **single ResNet-18 trained on 10,932 codepoint classes** using a synthetic corpus of ~5.47 M images (500 per class) engineered from three independent source families — system fonts, KanjiVG / e-hanja stroke medians, and Make-Me-a-Hanzi SVG strokes — and augmented on-GPU with perspective, blur, JPEG, neon glow, paper texture, and elastic warp. The class set comes from a custom **canonical variant database** (`sinograph_canonical_v2`) that reconciles variant families across Unicode, Unihan, KanjiVG, and e-hanja; this enables both strict top-1 accuracy and **family-aware accuracy** (credit if the prediction is a legitimate variant of the target). The project's novelty is two-fold. First, a **language-neutral class structure** that covers characters commodity OCRs treat as out-of-vocabulary — CJK Extension B (𤨒, 𤴡), Korean-only hanja (畓, 媤, 乶), and variant forms that language-mode OCRs normalize away. Second, a **targeted fine-tune workflow**: given a deployed base model, a user can push a chosen character's confidence in ~30 seconds of GPU time by generating 1 k fresh samples and running a 200-step mixed-batch fine-tune — far cheaper than the 4-hour full retrain — opening a per-user personalization path that OCR APIs do not expose.

---

## Key Features and Difficulty Justification

The key features are: a multi-source synthetic corpus that bypasses the dataset vacuum for rare characters; a GPU-resident style + augmentation pipeline (`synth_engine_v3`) that produces samples end-to-end at training speed; a ResNet-18 trained with AMP + cosine schedule + best-checkpoint tracking to **92.82 % val top-1 across 10,932 classes**; a **targeted fine-tune procedure** that reallocates confidence toward a chosen character without full retraining; and **family-aware evaluation** against the canonical variant DB. The project is difficult for four reasons. First, there is no off-the-shelf labeled dataset for 10,000+ balanced CJK classes — the class list, sources, and augmentation pipeline all had to be engineered from scratch. Second, training a 10,932-way classifier under a single-consumer-GPU budget required careful shard design (tensor shards, GPU-side resize) to hit throughput. Third, the **mutual-regression phenomenon** in targeted fine-tuning (boosting one character demonstrably suppresses its visual neighbors — 𤨒 ↔ 媤 trade confidence symmetrically in our experiments) is a real effect that constrains the fine-tune design. Fourth, even the evaluation required engineering — the only meaningful baseline is a running commodity OCR, so we built a side-by-side harness (`32_easyocr_compare.py`) that runs EasyOCR across multiple language groups and our classifier on the same images.

---

## Expected Results and Midpoint Status

By midpoint we have completed the corpus pipeline, trained the base classifier, and validated the targeted fine-tune procedure. On a curated 23-image benchmark (`train_engine_v*/test_img/`, codepoint-suffixed filenames), the base model scores **23 / 23 (100 %)** top-1 while EasyOCR (ch_sim + ch_tra + ja + ko + en) scores **10 / 23 (43 %)**. EasyOCR's 13 failures split cleanly into categories our class design targets: CJK Ext B → hallucinates BMP radicals (璁 / 琨 / 瑠 instead of 𤨒); Korean-only hanja → substitutes Chinese analog (媳 instead of 媤); stylistic stress (inverted / desaturated signage 中) → empty output. The **midpoint demo is a fixed test set of rare-character images where standard OCR fails or produces low-confidence output while our classifier returns correct top-1**; this is the same demo shape described in the original proposal, now backed by a quantitative baseline comparison. In addition, two fine-tune experiments (𤨒 → 91 % prob, and 媤 → 61 % on a real Korean sign, each in 26 s of GPU time) surface and quantify the mutual-regression failure mode, informing the remaining work. For the final demo, we will (1) widen Stage 1 augmentation to cover real-world signage photography (crop_bao / crop_xian class), (2) extend the fine-tune into a **multi-target co-training** variant that handles the 𤨒 ↔ 媤 zero-sum, and (3) package a minimal end-to-end CLI from image → top-5 prediction + variant family lookup. **Pi / Coral deployment is a stretch goal**: the ONNX + TFLite export path is scaffolded and will be shown if time permits, but the final demonstration of record is the classifier's accuracy delta against commodity OCR on a held-out test set — a reproducible Track-2 result that does not depend on hardware assembly.

---

## Appendix — Midpoint artifacts (reviewer-verifiable)

- Base model: `train_engine_v2/out/03_v3r_prod_t1/best.pth` (10,932 classes, ResNet-18 @ 128 px, 92.82 % val top-1)
- Corpus: `synth_engine_v3/out/80_production_v3r_shard256/` (5.47 M samples)
- Canonical variant DB: `sinograph_canonical_v2/out/sinograph_canonical_v2.sqlite`
- Targeted fine-tune experiments: `d:/tmp/24a12_exp/` (𤨒), `d:/tmp/5aa4_exp/` (媤)
- EasyOCR comparison harness: `train_engine_v2/scripts/32_easyocr_compare.py`
- Curated demo test set: `train_engine_v*/test_img/` (23 images, codepoint-suffixed filenames)
- Method document: `doc/15_TARGETED_FINETUNE_WORKFLOW.md`

# Lab 3 Project Context Document (v0.1)

## 1. Course Context

This document summarizes the current planning state for **Lab 3** in the IoT and Cognitive Computing course.

Lab 3 is an open-ended project where students must:

* Choose a project track (IoT System or DNN/Acceleration)
* Design and implement a working system
* Demonstrate the system at midpoint and final demo
* Show technical depth, novelty, and completeness

Evaluation focuses on:

* Completeness
* Quality
* Difficulty
* Novelty

The project should involve:

* Sensors or input devices (e.g., camera)
* Machine learning / DNN components
* A meaningful real-world problem
* A working demonstration

---

## 2. Initial Project Idea

### Theme:

**Rare and Region-Specific Chinese Character Recognition System**

The project idea originated from real-world frustration with OCR systems failing to recognize:

* Rare Chinese characters
* Korean-specific Hanja
* Stylized characters in games (e.g., BMS titles)
* Characters from historical sources (e.g., Kangxi dictionary scans)
* Variant glyph forms (異体字)

Example issue:

Some characters exist in Unicode databases but:

* Have no modern meaning
* Have region-specific meaning only
* Are rarely included in pretrained OCR models
* Fail recognition in existing OCR tools

Example:

Character:
**媤 (시집 시)**

This character:

* Exists in Unicode
* Has Korean meaning
* Is rarely recognized by standard OCR tools

---

## 3. Current Understanding of the Problem

The core issue is **not only OCR failure**, but the **long-tail nature of rare characters**.

Most OCR systems are trained on:

* Common modern characters
* Standard fonts
* High-frequency writing

However, failures occur in:

### Known failure scenarios:

1. Rare characters not included in training datasets
2. Variant glyph forms (異体字)
3. Stylized fonts (game UI fonts)
4. Historical scanned texts
5. Low-quality or noisy images
6. Region-specific characters with limited annotation

This suggests the existence of:

**Long-tail glyph recognition problems**

Where:

* A small set of characters is very common
* A very large set of characters is rare

---

## 4. Major Planning Uncertainties

The project idea is promising, but still uncertain in several ways.

### Key uncertainties:

#### A. Which domain to focus on?

Possible directions:

1. Game fonts (e.g., BMS titles)
2. Historical documents (e.g., Kangxi dictionary scans)
3. Korean-specific Hanja
4. Mixed rare-character domain

This decision will heavily affect:

* Dataset collection
* Model architecture
* Difficulty level

---

#### B. What exact failure mode to target?

Possible failure categories:

* Character not recognized at all
* Wrong character predicted
* Character segmentation failure
* Recognition confidence too low
* Variant glyph mismatch

Understanding this requires experiments.

---

#### C. What model design is actually needed?

Not yet determined.

Possible candidates:

* Character classifier (CNN-based)
* OCR fine-tuning
* Rare-character fallback model
* Similar glyph retrieval model

Decision depends on observed failure patterns.

---

## 5. Immediate Next Goal

**Reproduce failure cases using existing OCR tools.**

This is the most critical next step.

Without failure reproduction:

* No baseline exists
* No measurable improvement exists
* No meaningful DNN design can be justified

---

## 6. Planned Baseline Experiments

### Step 1 — Collect Test Samples

Target:

**30–100 character images**

Sources:

* Game titles (e.g., BMS)
* Kangxi-style dictionary scans
* Korean rare Hanja
* Variant glyph samples
* Some common characters (for comparison)

Each sample should include:

* Image
* Ground truth character (if known)
* Source type

---

### Step 2 — Run Multiple OCR Systems

Candidate OCR tools:

* EasyOCR
* Tesseract
* PaddleOCR

Each sample will be processed by all systems.

---

### Step 3 — Record Results

Create a table:

| ID | Image | Ground Truth | OCR Output | Confidence | Success | Notes |
| -- | ----- | ------------ | ---------- | ---------- | ------- | ----- |

Notes may include:

* Rare character
* Variant glyph
* Stylized font
* Low resolution

---

### Step 4 — Classify Failure Types

Define failure categories such as:

* Rare character out-of-vocabulary
* Variant mismatch
* Stylized font mismatch
* Low-quality scan
* Segmentation error

Goal:

Identify the **dominant failure mode**.

---

## 7. Expected Outcome of Baseline Stage

After baseline experiments:

We should know:

1. How often OCR fails
2. Where it fails
3. Why it fails
4. Which domain is most problematic

This will determine:

**Final project direction**

---

## 8. Preliminary DNN Direction (Not Final)

Possible directions after baseline:

### Option A — Rare Character Classifier

Train a CNN to recognize:

* Selected rare characters
* Variant glyphs

Input:

Single character image

Output:

Character label

---

### Option B — Fine-tuned OCR Model

Use pretrained OCR backbone and:

* Fine-tune on rare-character dataset
* Improve recognition on specific glyph domain

---

### Option C — Fallback Rare-Character Model

System design:

Standard OCR
→ If confidence low
→ Use specialized rare-character model
→ Lookup dictionary

This option is highly promising.

---

## 9. Potential System Architecture

Future possible structure:

Camera (Raspberry Pi)

↓

OCR Engine

↓

Confidence Check

↓

Rare Character Model (DNN)

↓

Character Database Lookup

↓

Display Result

---

## 10. Long-Term Vision

If successful, this system could:

* Recognize rare or historical characters
* Improve accessibility of East Asian texts
* Assist language learners
* Support digital humanities tools
* Help decode unknown glyphs

This gives strong motivation for the project.

---

## 11. Current Status Summary

Idea exists:
**Rare-character recognition assistant**

But still uncertain:

* Exact domain
* Dataset size
* Model type
* Difficulty level

Immediate priority:

**Baseline OCR failure reproduction**

Not model training yet.

---

## 12. Immediate Action Plan

Do this first:

1. Collect 30–100 sample images
2. Run 2–3 OCR tools
3. Record outputs
4. Classify failures
5. Identify dominant failure type

Then:

Design DNN solution.

---

End of Document (v0.1)

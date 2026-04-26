# Final Presentation — Sinograph Explorer

**Track 2 — Custom DNN Model**
Drafted: 2026-04-23. 7 slides total. Target talk length: ~8 minutes.

Structure: **Motivation → Workflow → DB → Synth → Training → Midpoint →
Final Goals**. Heavy detail only on Slide 5 (Training Engine), the rest
is brief scaffolding.

---

## Slide 1 — Motivation

**Commodity OCR is built for sentences, not isolated characters**

- Mainstream OCR (EasyOCR, Tesseract, Google Lens, Papago) is
  sentence-level by design: detect lines → recognize characters → apply
  a **language-model / dictionary prior** to smooth the output.
- That LM prior is what fails on **isolated rare glyphs**: no
  surrounding context to disambiguate → rare codepoints are
  "auto-corrected" toward visually similar common characters.
  - Korean-only `媤` → Chinese common `媳` (wrong meaning)
  - CJK Ext B `𤨒` → BMP common `璁` / `琨` / `瑠` (hallucination)
- This is a structural consequence of sentence-OCR design, not a
  training-data gap.
- We build a **single-character classifier** with **no LM**, so it
  commits to the pixel evidence — the complement to commodity engines
  on their weakness zone.

Visual: two EasyOCR failures (媤 → 媳, 𤨒 → 璁) with LM-prior arrow.

### Supporting visual — CJK Unicode block map

Where hanzi live on the Unicode plane, and where commodity OCR fails.

```
BMP (Basic Multilingual Plane)                   │  Commodity OCR
──────────────────────────────────────────────── │ ─────────────────
U+3400–U+4DBF   CJK Ext A           6,592 cp     │   works
U+4E00–U+9FFF   CJK Unified        20,992 cp     │   works (common)
U+F900–U+FAFF   CJK Compat Ideos      472 cp     │   inconsistent
                                                 │   (homoglyph of
                                                 │    Unified block)
──────────────────────────────────────────────── │ ─────────────────
SMP (Supplementary Multilingual Plane)           │
──────────────────────────────────────────────── │ ─────────────────
U+20000–U+2A6DF  Ext B              42,718 cp    │   ✗ hallucinate
U+2A700–U+2B73F  Ext C               4,149 cp    │   ✗
U+2B740–U+2B81F  Ext D                 222 cp    │   ✗
U+2B820–U+2CEAF  Ext E               5,762 cp    │   ✗
U+2CEB0–U+2EBEF  Ext F               7,473 cp    │   ✗
U+2EBF0–U+2EE5F  Ext I                 622 cp    │   ✗
U+2F800–U+2FA1F  Compat Ideos Supp     542 cp    │   ✗
U+30000–U+3134F  Ext G (2020)        4,939 cp    │   ✗
U+31350–U+323AF  Ext H (2022)        4,192 cp    │   ✗
U+323B0–U+33479  Ext J (2025, new)   4,298 cp    │   ✗ no fonts yet
```

**Headline takeaways**

1. **Common hanzi cluster in U+4E00–U+9FFF (BMP 20 k)** — everyday
   Chinese/Japanese/Korean text lives here. ~3,500 codepoints cover
   99 % of modern daily usage. Commodity OCR targets this band.
2. **SMP Ext B–J (~75 k codepoints) is where commodity OCR fails** —
   rare / historical / dialectal / newly-standardized hanzi. No
   language-model vocabulary contains them, so sentence-OCR substitutes
   them with visually similar BMP common chars (the "hallucination"
   pattern in our EasyOCR comparison).
3. **BMP contains Korean-only hanja** — `畓 U+7553`, `媤 U+5AA4`,
   `乶 U+4E76`, `垈 U+5788`, etc. Technically in the common band, but
   commodity OCR's Chinese-dominated LM prior pulls them toward the
   look-alike Chinese common char (`媤 → 媳`).
4. **CJK Compat block (U+F900–U+FAFF) is a homoglyph trap** — same
   glyph shape with a different codepoint, kept for legacy KS X 1001 /
   JIS round-trip. NFKC normalizes these; commodity OCR does it
   inconsistently.

**Our target coverage**

| Scope | Codepoints | Source |
|---|---:|---|
| Midpoint baseline (trained) | **10,932** | T1 subset (BMP-centric) |
| canonical_v3 database (labels ready) | **103,046** | BabelStone ∪ CHISE ∪ cjkvi, full SMP |
| Final demo target (training plan) | **~76,000** | e-hanja online universe — the Korean-weighted intersection of BMP + Ext A/B + Compat |

The final-demo goal is to push the trained class set from the
BMP-biased 10,932 to the **76 k e-hanja-weighted set**, so the model
keeps its commodity-OCR advantage on BMP and **adds the ~42 k SMP Ext
B–J codepoints that commodity engines cannot produce at all**.

---

## Slide 2 — Overall Workflow

```
  ┌─────────────────────┐
  │ Building            │   reverse-engineer / ingest 10+ CJK
  │ Canonical DB        │   dictionaries → per-codepoint labels
  └──────────┬──────────┘   (radical / strokes / IDS / family / readings)
             │
  ┌──────────▼──────────┐
  │ Building            │   GPU pipeline: fonts + SVG strokes +
  │ Synth Engine        │   randomized augmentation → training images
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │ Building            │   structure-aware multi-task ResNet-18
  │ Training Engine     │   (char + radical + stroke + IDC heads)
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │ Compare with        │   image benchmark, family-aware accuracy,
  │ Existing OCR        │   confusable-pair stress test, latency
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │ Deploy to Pi        │   ONNX → INT8 TFLite
  │                     │   Pi TPU/CPU inference
  └─────────────────────┘
```

Each block is owned by one sub-engine / one folder in the repo.

---

## Slide 3 — Building the Canonical DB

Despite not being the Track-2 contribution directly, the DB is the
**prerequisite for structure-aware training**. Without it, each hanzi
has only one label (its codepoint). With it, many hanzi additionally
get **radical / stroke / IDS / variant-family** supervision → the
model can exploit structural redundancy across classes, which raises
accuracy on confusable and rare characters.

**Reverse-engineering**: Korean e-hanja online dictionary had no public
API; scraped SVG + JSON + HTML endpoints, classified animated vs
static, cleaned watermark, normalized stroke SVG — **one short pipeline**
documented in `db_mining/RE_e-hanja_online/`.

**Sources ingested by country**:

| Country / origin | Source | Scale |
|---|---|---:|
| International | **Unihan** (Unicode backbone) | ~97 k |
| International | **BabelStone IDS** (region-tagged trees) | 97,649 |
| International | **CHISE IDS** (GPLv2, widest coverage) | 102,892 |
| International | **cjkvi-ids** (CHISE-derived, Korean-aligned) | 88,937 |
| Korea | **e-hanja online + mobile** | 76,013 |
| China | **MakeMeAHanzi** (decomposition + stroke graphics) | 9,574 |
| China / Taiwan | **CNS11643** (대만 全字庫) | ~70 k |
| Taiwan | **MOE Revised Dict / MOE Variants** (교육부) | ~100 k |
| China (mainland) | **Tongyong Guifan** (통용규범한자표) | ~8 k |
| Japan | **KanjiVG** (SVG strokes) | 6,699 |
| Japan | **KANJIDIC2** (일본 상용·인명용) | ~13 k |

Useful fields (per codepoint):
- **radical (1–214)** — 강희자전 classical radical index
- **total_strokes** + **residual_strokes** (= total − radical strokes)
- **IDS decomposition tree** (`⿰金監`, `⿱水田`, …)
- **IDS top-level IDC** (12-way layout: ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻)
- variant family + canonical representative (for evaluation)
- readings (한국 훈음 / mandarin / japanese / vietnamese / cantonese)
- meanings (Korean / English / Chinese)

Merged table `characters_ids` = **103,046 codepoints**, `characters_structure`
at **99.9 %+ coverage** on 4 training aux labels. (Details in
[sinograph_canonical_v3/BUILD_STATUS.md](../sinograph_canonical_v3/BUILD_STATUS.md).)

---

## Slide 4 — Building the Synth Engine

Per-class training images are **synthesized**, not collected — no public
labeled dataset covers 10 k+ balanced CJK classes.

**Pipeline** (`synth_engine_v3`):
- Glyph source: system fonts + KanjiVG SVG strokes + e-hanja medians +
  MakeMeAHanzi SVG
- Randomized augmentation applied on GPU per sample:
  perspective / rotation / Gaussian blur / JPEG compression / chromatic
  aberration / neon glow / paper texture / elastic warp / color jitter
  / inversion / noise / binarization / saturated-background overlay
- Output: tensor shards (`.npz`), ~1 k samples per shard,
  pre-resized 256² (decoded on-GPU to 128² for training)
- Production run: **500 samples × 10,932 classes = 5.47 M images**,
  generated in ~4 h on a single GPU

The randomization matters — every class sees dozens of font families,
sign-like color schemes, stroke-weight variations, degradation levels,
in both dark-on-light and light-on-dark modes.

Visual: 8×8 grid of randomly augmented samples from a single shard
(existing `d:/tmp/shard_demo/grid_8x8.png`).

---

## Slide 5 — Building the Training Engine

### The challenges specific to hanzi OCR

Vanilla cross-entropy on codepoint labels alone has characteristic
failure modes we measured at midpoint:

1. **Fine-grained confusion among same-radical neighbors** — 鑑 top-1 =
   35.6 %; top-2…top-5 are all 金-radical metal characters.
2. **Low confidence on rare or stylized inputs** — `𤨒` on a clean font
   render scores only 40 %; `媤` on a real Korean sign scores 4.6 %.
3. **Mutual regression between visual neighbors** — targeted fine-tune
   of `𤨒` drops `媤` from rank 1 to rank 6, and vice versa.
   Visually adjacent classes trade confidence zero-sum.
4. **Compositional blindness** — the model cannot exploit the fact that
   hanzi decompose into (radical + phonetic component + IDC layout) as
   a design principle. It must rediscover this from pixels alone.

### Our positioning — where the novelty lives

We do **not** claim a new backbone or a new conv block. The novelty is
in **supervision design for CJK recognition**: a canonical DB provides
structural labels that do not exist in any public dataset, and the
loss / head topology is built around how hanzi actually compose, not
around image-classification orthodoxy. Concretely we stage three
levels of structural supervision, **A → A+ → B**, each strictly more
compositional than the last. Cost at inference stays the same — all
aux heads discard at deploy.

| Level | Supervision added | Status | Inference cost | Why it matters |
|---|---|---|---|---|
| **A** | 4 aux labels (radical / total strokes / residual strokes / IDC) | **primary plan** — labels 99.9 %+ ready | +0.7 % params, 0 ms latency | teaches "features a human uses" (radical, stroke count, layout) |
| **A+** | component multi-label + confusable-pair margin | **add if Level A accuracy plateaus** | +K·512 params (K ≈ 512), 0 ms | turns recognition **partially compositional**; directly attacks 鑑/媤-style failures |
| **B** | radical-conditioned scoring fusion | **stretch goal** | 0 extra params (score fusion) | hierarchical like a dictionary: `p(char|x) = p(radical|x)·p(char|radical,x)` |

---

### Level A — structure-aware multi-task (primary plan)

Inject CJK writing-system structure as **auxiliary supervision**, not
as hard-coded rules. Shared backbone must produce features that are
simultaneously useful for all 5 predictions → representation is forced
to encode compositional structure.

**Input**: RGB `128 × 128 × 3`, normalized `(pixel/255 − 0.5)/0.5` → `[−1, 1]`.

**Backbone — torchvision `ResNet-18`, `weights=None`, trained from scratch**:

| Stage | Spec | Output `C × H × W` | Params |
|---|---|---|---:|
| `conv1` | 7×7 conv, 64 ch, stride 2 + BN + ReLU | 64 × 64 × 64 | 9.4 k |
| `maxpool` | 3×3 stride 2 | 64 × 32 × 32 | — |
| `layer1` | 2 × BasicBlock (64 ch) | 64 × 32 × 32 | 148 k |
| `layer2` | 2 × BasicBlock (128 ch, first s=2) | 128 × 16 × 16 | 526 k |
| `layer3` | 2 × BasicBlock (256 ch, first s=2) | 256 × 8 × 8 | 2.10 M |
| `layer4` | 2 × BasicBlock (512 ch, first s=2) | 512 × 4 × 4 | 8.39 M |
| `avgpool` | adaptive 1×1 | 512 × 1 × 1 | — |

BasicBlock = `conv3×3 + BN + ReLU + conv3×3 + BN + residual + ReLU`.
Total: 17 conv + 1 fc = **ResNet-18**. Backbone = **11.18 M params**.

**Five parallel Linear heads on the 512-d feature**:

| Head | Layer | Output | Params | Loss | Weight |
|---|---|---|---:|---|---:|
| **char** (primary) | `Linear(512 → 10,932)` | 10,932 logits | 5.61 M | CE | 1.0 |
| **radical** (aux) | `Linear(512 → 214)` | 214 logits | 110 k | CE | 0.2 |
| **total_strokes** (aux) | `Linear(512 → 1)` | scalar | 0.5 k | MSE | 0.1 |
| **residual_strokes** (aux) | `Linear(512 → 1)` | scalar | 0.5 k | MSE | 0.1 |
| **ids_top_idc** (aux) | `Linear(512 → 12)` | 12 logits | 6 k | CE | 0.2 |

**Loss**:
`L_A = CE(char) + 0.2·CE(radical) + 0.1·MSE(total) + 0.1·MSE(residual) + 0.2·CE(IDC)`

**Model totals**: 16.78 M (baseline) → **16.90 M multi-task** = **+117 k /
+0.70 %**. Aux heads are **discarded at inference** → same latency as
baseline (3–5 ms GPU, ~150 ms Pi CPU).

Why this design: radical, stroke count, and IDC are **exactly the
features a human would use to distinguish 未/末, 田/由/甲/申, or
媤/媳**. The net is forced to attend to them explicitly, rather than
hoping pixel-only training discovers them.

---

### Level A+ — compositional + confusable-aware supervision (planned)

Level A still treats each class as a single softmax atom. Level A+
upgrades this in two ways — **what a character is made of** and
**what it is easy to confuse with**. Both are drop-in additions that
require no backbone change and cost 0 ms at inference.

**(1) Component multi-label head** — each codepoint carries a
multi-hot vector over a shared component vocabulary (e.g. 鑑 →
`{金, 監}`). A top-K frequency vocabulary (K ≈ 512) is mined from
canonical_v3 `ehanja_components_json` (76 k cp) ∪ IDS flat-component
list (103 k cp). See [17_CANONICAL_V3_PLAN.md](17_CANONICAL_V3_PLAN.md)
Phase 1.5.

```
component_head: Linear(512 → K)
L_component = BCEWithLogitsLoss(positive-weighted, K ≈ 512)
```

**Why**: the char head alone learns codepoint identity; the component
head forces the backbone to additionally encode **part identity**.
This is where pixel-only training fails on 鑑 / 鍳 / 鐱: they share
金 as a component but the flat softmax has no mechanism to exploit
that relationship.

**(2) Confusable-pair margin loss** — after Level A is trained once,
mine hard pairs from the validation confusion matrix and from a
heuristic (same radical + stroke-count Δ ≤ 1). Impose a margin so the
true class logit beats the worst confusable negative by at least *m*:

```
L_margin = max(0, m − z_y + max_{c ∈ confusable(y)} z_c)
```

**Why**: our midpoint data shows the failure is not spread uniformly
across 10 k classes — it is concentrated on a few hundred visually
adjacent pairs (鑑/鍳, 媤/媳, 未/末, 𤨒/璁). A margin loss targets
exactly those pairs instead of reweighting the whole softmax.

**Level A+ loss**:
`L_A+ = L_A + 0.2·L_component + 0.15·L_margin`

---

### Level B — radical-conditioned hierarchy (stretch goal)

Replace the flat 10 k-way softmax scoring with a dictionary-style
decomposition:

```
score(c | x) = z_char[c] + α · log p(radical(c) | x)
p(char | x) = softmax(score)
```

The radical head from Level A is reused as a conditioning signal on
the char logits. **Zero new parameters** — this is a scoring-time
fusion, not a new head. If the fused scoring lifts accuracy, upgrade
to per-radical sub-classifiers (214 local heads).

**Why**: people read hanzi hierarchically — first the radical, then
the rest of the character inside that radical's namespace. Dictionaries
are organized this way. Flat softmax ignores this structure entirely.

---

### Roadmap summary — why A → A+ → B, in this order

- **A first** because the 4 aux labels are already at 99.9 %+ coverage
  and the implementation is already scoped in `train_engine_v2`.
  Establishes the baseline structure-aware accuracy number.
- **A+ second** because it is (a) low-risk additive (no backbone
  change), (b) directly attacks the midpoint failure modes (鑑, 媤,
  𤨒 mutual regression), and (c) uses labels that canonical_v3 already
  has — only a small vocabulary-mining step is new.
- **B last** because scoring fusion is trivial to add once radical
  accuracy is trustworthy, but per-radical local heads are an
  implementation rabbit hole that only pays off after A+ stalls.

The three levels compose: each keeps the previous level's weights and
aux heads. There is no "throw away Level A to build A+" transition.

---

## Slide 6 — Midpoint Performance

**Baseline model** (midpoint submission, vanilla ResNet-18 without aux
heads, trained on 10,932 classes):

- Synthetic val top-1: **92.82 %** after 17 epochs on a single RTX
  4070 Ti
- 23-image real-world benchmark: **23 / 23 (100 %)** vs EasyOCR
  10 / 23 (43 %)
- **Latency measured on dev machine (Windows + GPU/CPU)**:
  - GPU: 3–5 ms per character
  - CPU: ~50 ms per character
- **Deployment verified**: `.pth → ONNX (63 MB) → INT8 TFLite (16 MB,
  4× compression)`. ONNX ↔ Torch parity 7 × 10⁻⁶. **TFLite produced
  but Pi inference not yet measured** — Windows TFLite XNNPack has a
  bug that blocks local validation; the conversion itself is verified
  correct.
- Targeted fine-tune workflow: `𤨒` confidence 40 % → 91 % in 26
  seconds; `媤` on a real Korean sign 4.6 % → 61 %. Also surfaced the
  **mutual-regression phenomenon** (𤨒 ↔ 媤 trade confidence).

What's done and what isn't: ✅ corpus / baseline / midpoint demo /
targeted FT / ONNX+TFLite export; ❌ Pi-side inference timing, Pi-side
accuracy verification.

---

## Slide 7 — Final Demo Goals

**Target — accuracy that matches the e-hanja online coverage** (76 k
Korean-aware codepoints) with the structure-aware multi-task model.
Coverage ambition: extend the 10,932-class baseline to the ~76 k cp
label set that canonical_v3 already supports.

**What we will show at the final demo**

1. **Pi-hardware inference** — actual per-character latency measured on
   the Raspberry Pi (CPU, optionally Coral TPU) with the INT8 TFLite
   model. Target: confirm real-time (< 500 ms per character) on Pi
   CPU, < 50 ms with Coral.
2. **Accuracy lift vs midpoint** — structure-aware model evaluated on
   (a) the 23-image benchmark (ceiling already hit at 100 %),
   (b) a new confusable-pair stress set (未/末, 田/由/甲/申,
   己/已/巳, 媤/媳, …) where the midpoint baseline is known to
   struggle, and (c) family-aware accuracy on a held-out test set.
3. **End-to-end demo** — photo → top-1 codepoint → canonical_v3 lookup
   → Korean reading + meaning + variant family displayed. Running on
   the Pi.

Closing one-liner: *"Commodity OCR handles the head. We handle the long
tail — with a database and a loss that know the writing system, on a
Raspberry Pi."*

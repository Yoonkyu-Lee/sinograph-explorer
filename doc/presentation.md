---                                                                                                                                                                               
  발표 대본 (Target ~5 min, English)                                                                                                                                                
                                                                                                                                                                                    
  Slide 1 — Title (~10s)                                          

  ▎ "Good morning everyone. My project is Sinograph Explorer — Track 2, Custom DNN Model. I'm Yoonkyu Lee."

  ---
  Slide 2 — Problem Statement (~30s)

  [rubric: Why important? + What are existing solutions?]

  ▎ "Chinese, Japanese, and Korean writing systems share a common pool of logographic characters — I'll call them 'sinographs' for short. Unicode — the international standard that
  ▎ assigns every character a unique numeric codepoint — splits them into the BMP, the Basic Multilingual Plane where everyday characters live, and the SMP, where the rare ones
  ▎ live: classical, historical, dialectal, and newly-standardized characters. OCR — optical character recognition, the task of turning images of text back into characters — is
  ▎ what we'd normally reach for here. Commodity engines like Tesseract, Google Lens, even modern multimodal LLMs, handle BMP fine, but systematically fail on SMP. For scholars
  ▎ working with classical texts, ancient inscriptions, or regional variants like Korean Hanja or Vietnamese chữ Nôm, that gap is real."

  ---
  Slide 3 — Project Goal (~25s)

  [rubric: Why important?]

  ▎ "So my goal is to build an OCR that reads these characters — including CJK Extension B, variant forms, and the characters commodity engines miss. Final deployment target:
  ▎ Raspberry Pi with Coral Edge TPU, so it runs offline at the edge."

  ---
  Slide 4 — Overall Workflow (~20s)

  [rubric: What does your design look like?]

  ▎ "Here's the full pipeline. Three stages: a Canonical DB that holds structural metadata for every codepoint, a Synth Engine that renders training images, and a Training Engine 
  ▎ that produces the deployable model. The final artifact ships to the Pi."

  ---
  Slide 5 — Canonical DB (~35s)

  [rubric: Each part with technical details]

  ▎ "A quick backstory on the Canonical DB — I originally built it for Track 1, as the lookup index for a sinograph dictionary app, so it already had structured metadata per
  ▎ codepoint. When I switched to Track 2, that same metadata turned out to be exactly what a structure-aware DNN needs as auxiliary labels — the DB came along for free."
  ▎
  ▎ "For every codepoint I store: the classical radical out of 214, total and residual stroke counts, the IDS decomposition tree — for example 鑑 decomposes as ⿰金監 — and the
  ▎ top-level IDC operator out of 12 layout primitives. This is what lets the rest of the system be structure-aware instead of pixel-only."

  ---
  Slide 6 — Synth Engine (~25s)

  [rubric: Each part + Challenges]

  ▎ "The Synth Engine renders each character from fonts and SVGs, then applies heavy randomized augmentation — perspective, blur, JPEG artifacts, paper texture, elastic warp, color
  ▎  jitter — to mimic real-world capture. At midpoint I trained on about 11k classes; the final corpus is 102k classes × 200 samples each, around 20 million images."

  ---
  Slide 7 — Training Engine input (~20s)

  [rubric: Each part + Why this way?]

  ▎ "The model is ResNet-18 from scratch, on 128×128 RGB normalized to negative one through one. Light enough to compile to Edge TPU later, but deep enough for the structure
  ▎ features."

  ---
  Slide 8 — Baseline 5-head loss (~45s)

  [rubric: Each part + Results — first concrete number]

  ▎ "For the midpoint baseline I used a 5-head joint loss: cross-entropy on character ID, plus auxiliary losses on radical, total strokes, residual strokes, and IDC layout. The
  ▎ intuition: radical, stroke count, and IDC are exactly the cues a human uses to distinguish look-alikes like 未 versus 末 or 田/由/甲/申. Forcing the backbone to predict them
  ▎ explicitly should help."
  ▎
  ▎ "I trained this on the full 102k-class corpus and got 38.99% top-1. That's my baseline number."

  ---
  Slide 9 — Pivot to SCER ⭐ (~70s)

  [rubric: What do you do to improve? + Benefit + Why this way? — most important slide]

  ▎ "After midpoint I realized the two ideas on this slide — a component head and a confusable-pair margin — are solving the same problem: how to position each character in some
  ▎ space. So I unified them."
  ▎
  ▎ "I replaced the linear 102k-way softmax with a 128-dimensional L2-normalized embedding head, trained with ArcFace — an angular-margin classifier with margin 0.5, scale 30.
  ▎ ArcFace pulls same-class samples together and pushes different-class samples apart. That's the confusable-pair margin idea, but applied uniformly to every pair, not a
  ▎ hand-picked list."
  ▎
  ▎ "At inference, the ArcFace classifier weights become an anchor database — one 128-dim anchor per class. A query image just gets compared by cosine similarity. And here's where
  ▎ the structure heads come back in: instead of being a training-time auxiliary, they become a deploy-time prefilter — radical top-3, IDC top-2, strokes within plus-minus two —
  ▎ narrowing 102k candidates down to a few hundred before cosine search."
  ▎
  ▎ "Two benefits: it's faster at deploy, and adding a new character means adding one anchor — no retraining."

  ---
  Slide 10 — Demo Goals + Status (~50s)

  [rubric: Results + Demo + Good/improve / Bad/adjust]

  ▎ "For the final demo, three deliverables."
  ▎
  ▎ "One — Pi-hardware latency. Measuring per-character inference time on the Coral TPU with the INT8 TFLite model. The Keras port and INT8 compile are already done; what's left is
  ▎  on-device measurement."
  ▎
  ▎ "Two — structure-aware versus softmax. SCER versus the 38.99% baseline, evaluated on a confusable-pair stress set: 未/末, 田/由/甲/申, 己/已/巳, 媤/媳, and so on. SCER is in
  ▎ production training right now, about 17 hours wall-clock; the smoke run hit all six loss components healthy."
  ▎
  ▎ "Three — if time permits, live capture with a Pi camera."
  ▎
  ▎ "Honest status: training-side design is implemented and validated. The open piece is on-device latency, which is hardware-gated. Demo next week."

  ---
  Closing (~5s)

  ▎ "Thank you — happy to take questions."

  ---
  시간 조절 가이드

  ┌──────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │     분량     │                                                                    어디서 줄이고 / 늘리나                                                                    │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 5 min 빡빡   │ Slide 5, 6, 7 각각 한 문장씩 빼기. Slide 9 의 "Two benefits" 부분만 유지                                                                                     │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 5:30 적당    │ 위 그대로                                                                                                                                                    │
  ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 8-10 min     │ Slide 9 에 "ArcFace 가 random init 에서 m=0.5 로 시작하면 발산해서 m curriculum 도입했다" 같은 디테일 추가, Slide 10 에 throughput number (3,120 img/s,      │
  │ 여유         │ batch 640) 추가                                                                                                                                              │
  └──────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Q&A 방어 라인 (예상 질문)

  - "Why ArcFace and not triplet?" — "Triplet needs explicit pair mining; ArcFace gets the same effect from a single CE-style loss with the classifier weights as anchors. Cleaner
  training signal."
  - "What's your top-1 with SCER?" — "Production run is in progress, so I'll have the number for the demo. Smoke-scale is already above the random baseline; the gate I care about
  is cosine-NN top-1 on the confusable stress set, not raw top-1."
  - "Why ResNet-18, not something bigger?" — "Edge TPU constraint. Coral TPU has 8 MB on-chip SRAM; bigger backbones page off-chip and lose the latency benefit."
  - "Augmentation list looks heavy — overfitting risk?" — "Opposite — 102k classes × 200 samples is under-sampled per class. Augmentation is what makes 200 samples informative."
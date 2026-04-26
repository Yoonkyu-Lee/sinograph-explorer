# Stage 2 Training — 실전 설계 문서

Stage 1 (이미지 생성) 과 나란히 읽는 문서. Stage 1 = `doc/10`, Stage 2 = 본 문서.

이 문서의 목적은 **이미지 생성이 끝나는 순간 바로 학습을 돌릴 수 있도록 "뭐를 준비해두면 되는지"** 를 좁혀서 적어두는 것. 엔진 구현에 비해 학습 쪽은 "뭘 안 해도 되는지" 를 명확히 해야 시간이 안 새는 단계임.

---

## 0. 평가 기준 재확인 (lec17)

Lab3 rubric 은 4 축: **Completeness / Quality / Difficulty / Novelty**.

- Proposal 은 Track 1 (Pi + Coral) 로 명시했지만, lec17 Track 2 "software approach = design and optimize novel ML algorithms" 조건도 충족하므로 **demo 프레이밍을 Track 2 쪽으로 기울여서** Novelty 축을 살리는 쪽으로 간다.
- Coral TPU 컴파일은 **선택 bonus** — 시간 남을 때만. Lab2 에서 이미 FaceNet → tflite → Coral 경로를 통과했으므로 "repeat" 성격이고, 평가 가치 낮음.
- 이 프로젝트의 진짜 novelty 는 이미 **Stage 1 의 multi-source 생성엔진 + canonical DB v2 의 variant graph** 에 집중되어 있음. Stage 2 는 그 두 자산을 "ML 알고리즘으로 활용하는 단" 이다.

### 마감선
- **Midpoint demo: 4/20 (월, 내일)** — preliminary 만 있으면 통과. 코드가 돌고 loss 가 내려간다는 증거면 충분.
- **Final demo: 5/4 (월)** — 전체 학습 + 평가 + 보고서.

---

## 1. 학습 워크플로 한 눈에

```
  [Stage 1 산출물]                      [Stage 2 학습]                        [평가 / 내보내기]

  synth_engine_v3/out/                    train_engine/                        report/
   42_production_mobile/                    scripts/                            figures/
   ├─ 000000_U+4E00_malgun-0.png             20_train.py       ─ loss curves
   ├─ 000001_U+4E8C_gulim-2.png              21_eval.py        ─ top-1/5 table
   ├─ ...                                    22_export.py      ─ confusion mat
   └─ corpus_manifest.jsonl     ─────►     configs/                            family-aware acc
                                              resnet18_t1.yaml                  (선택) tflite
         ▲                                  modules/
         │                                    dataset.py    ── manifest 읽기 + class index
    sinograph_canonical_v2/                  model.py      ── ResNet-18 factory
     canonical_v2.sqlite       ─────►        train_loop.py ── AMP + class-balanced sampler
     (variant graph 조회용)                   eval_family.py── DB lookup → family-aware
```

흐름:
1. **manifest.jsonl 읽기** → `sorted(unique(notation))` = class index
2. **PyTorch `Dataset`** 하나 — `__getitem__` 에서 PNG 로드 + notation → class_idx
3. **DataLoader** (class-balanced sampler 는 T1-only 면 실질적 불필요, 500/class 균등)
4. **ResNet-18** (from scratch, 10,932 클래스 FC layer)
5. **Train loop** — AMP (fp16), cosine LR, label smoothing 0.1
6. **Eval** — standard top-1/5 + variant family-aware (DB 조회)
7. (선택) **Export** — ONNX → TFLite INT8 → edgetpu_compiler

---

## 2. "진짜 필요한 것" vs "뒤로 미뤄도 되는 것"

### 진짜 필요 (midpoint 이전 or final 필수)

| 항목 | 이유 | 언제 |
|---|---|---|
| **manifest 기반 `Dataset`** | 파일 이동 없이 즉시 학습 가능. doc 10 § "라벨을 끌어 쓰는 세 가지 방법 (B)" | midpoint |
| **파일럿 학습 loop** (mini 50~100 클래스, 2k 샘플) | midpoint 에서 "코드는 돈다" 증빙 — loss 곡선 한 장 | midpoint |
| **전체 학습 loop** (10,932 클래스, 5.47M 샘플, ResNet-18) | final 메인 결과 | final |
| **평가: top-1 / top-5 on font-unseen split** | 기본 분류 성능 증거 | final |
| **평가: variant-family-aware accuracy** | proposal 의 "language-independent identity" novelty 를 양적 증거로 | final |

### 선택 (시간 남으면)

| 항목 | 이유 | 우선도 |
|---|---|---|
| Ablation (single-source vs multi-source 학습 비교) | "why synth engine matters" 를 ablation 으로 증명 | 중 |
| ONNX 변환 + PC-side INT8 latency 벤치 | Track 1 보험, 보고서에 숫자 하나 더 | 중 |
| Coral TPU edgetpu_compile + Pi inference 데모 | Track 1 원래 계획. 컴파일 op 지원 이슈 시간 많이 먹음 | 낮음 |
| Contrastive / triplet head (variant graph 를 positive pair source 로) | 추가 novelty. 기본 CE 가 먼저 돌고 나서 | 낮음 |

### 굳이 하지 말 것

- **mixup / cutmix / RandAugment** — 이미지 단에서 이미 v3 augment pipeline 이 25 op × 확률적 적용 중. 학습 단에서 더 augment 는 double-warp 위험.
- **학습 시점의 추가 randaugment** — 같은 이유.
- **사전학습 backbone (ImageNet pretrained)** — 한자 glyph 은 ImageNet 의 자연이미지 분포와 너무 멀음. From scratch 가 깨끗하고, proposal 의 "라벨 노이즈 0" 주장과도 정합적.
- **DDP / 멀티 GPU** — RTX 4080 하나뿐. 세팅 시간 0.

---

## 3. Phase 분할 (Stage 1 과 같은 스타일)

### S2-A — 학습 코드 스켈레톤 (midpoint 전, 수시간)

- `train_engine/` 워크스페이스 생성
- `modules/dataset.py`, `modules/model.py`, `modules/train_loop.py`, `scripts/20_train.py` 스켈레톤
- **50~100 클래스 mini subset** (production run 진행 중 생긴 manifest 앞쪽 2k row 만 slice) 로 loss 수렴 확인
- 목적: **midpoint 슬라이드에 loss curve 한 장 + "코드 존재" 증빙**. accuracy 숫자는 아직 의미 없음.
- 실행 시간 예상: 코드 30분 + mini 학습 10분 = 40분

### S2-B — 파일럿 수렴 (production run 도중 or 완료 직후)

- 본격 training loop 를 manifest 서브셋 (500 클래스 × 500 샘플 = 250k) 에 적용
- 목적: "class 당 500 샘플이 수렴에 충분한가" 를 경험적으로 확인. plateau 하면 full 에서도 500 으로 감.
- 실행 시간 예상: 30분~1시간 (RTX 4080, batch 256, ResNet-18 @ 128²)

### S2-C — Production 학습 (final 전)

- 전체 10,932 클래스 × ~500 샘플 = 5.47M
- ResNet-18 @ 128² 해상도 (256² 대비 4× 빠름, glyph 특성상 해상도 손실 영향 적음 — Phase A 에서 검증 필요하나 안전한 기본값)
- batch 256 + AMP (fp16)
- 20 epoch — **추산: ~2일** (5.47M / 256 = 21k step/epoch, 4080 에서 ResNet-18 fwd+bwd ~3 step/s with AMP → epoch ≈ 1.9 h → 20 epoch ≈ 38 h). 필요시 10 epoch 로 조기 종료하고 final demo 직전 재개.
- 중단/재개 가능한 checkpoint 저장 (`ckpt_epoch_{N}.pth`)

### S2-D — 평가 + 보고서용 산출물 (final 직전)

1. **Top-1 / Top-5 on font-unseen validation split**
   - manifest 의 `picked_source` 로 특정 font face 를 val 전용으로 분리 (예: `malgun-*` 은 val, 나머지 train). 단순하고 설명 가능한 split.
2. **Family-aware accuracy**
   - 예측 codepoint 가 정답과 같은 canonical_v2 variant family 안이면 partial credit (예: 0.5). 제안서의 "language-independent identity" 를 정량화.
   - `sinograph_canonical_v2/canonical_v2.sqlite` 에서 variant family 조회.
3. **Confusion 분석 top-20 실패 케이스**
   - 어떤 pair 에서 혼동이 잦은지 이미지 grid 로. 보고서 diagnostic 섹션용.
4. (선택) **Single-source ablation**
   - 같은 backbone 을 font-only 서브셋으로만 학습 → 동일 val split 에서 top-1 비교. multi-source 의 기여도 증명.
5. (선택) **ONNX + TFLite INT8**
   - calibration set = per-class 4 샘플 × 10,932 = 43k 이미지 서브셋
   - PC 상 INT8 latency 벤치 숫자만 확보

---

## 4. 파일 / 폴더 레이아웃 (제안)

```
train_engine/
├── README.md
├── configs/
│   ├── resnet18_t1_pilot.yaml         # S2-A/B 용, mini subset
│   └── resnet18_t1_full.yaml          # S2-C production
├── modules/
│   ├── __init__.py
│   ├── dataset.py                    # CorpusDataset(manifest_path, class_index, transform)
│   ├── model.py                      # build_resnet18(num_classes)
│   ├── train_loop.py                 # train_one_epoch, evaluate
│   ├── family_eval.py                # canonical_v2 SQLite 조회 + partial credit
│   └── utils.py                      # logging, checkpointing
├── scripts/
│   ├── 20_train.py                   # CLI: --config, --resume, --out
│   ├── 21_eval.py                    # CLI: --ckpt, --split val
│   ├── 22_export_onnx.py             # (선택) ONNX export
│   └── 23_quantize_tflite.py         # (선택) INT8 calibration
└── out/
    ├── 01_pilot_mini/                # S2-A 결과
    ├── 02_pilot_500cls/              # S2-B 결과
    └── 03_prod_t1/                   # S2-C 결과
        ├── ckpt_epoch_10.pth
        ├── metrics.jsonl
        └── confusion_top20.png
```

(`CLAUDE.md` 규칙: CLI entry 는 `NN_name.py` prefix, library 모듈은 prefix 없이.)

---

## 5. 코드 스케치 — 제일 헷갈리는 부분만

### 5.1 Dataset

```python
# modules/dataset.py
class CorpusDataset(Dataset):
    def __init__(self, manifest_path, split="train", transform=None, val_sources=None):
        self.samples = []          # list of (filename, class_idx)
        self.classes = []          # list of notation, sorted
        # manifest 1-pass
        # 1) 모든 notation unique 모아 sort → self.classes
        # 2) 샘플마다 filename + class_idx(= index into self.classes)
        # 3) val_sources 가 None 이면 random 10% split, 아니면 picked_source 로 필터
        ...
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fn, cls = self.samples[idx]
        img = Image.open(fn).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, cls
```

**중요**: manifest 1-pass 로 class index 를 고정한다. 이 인덱스는 ckpt 와 묶여 있어야 하므로 학습 시점에 `out/03_prod_t1/class_index.json` 으로 dump → export / eval / inference 시점 재사용.

### 5.2 Transform (학습 시)

v3 augment 가 이미 강하므로 학습 시점 transform 은 **최소한**:

```python
transform_train = Compose([
    Resize(128),              # 256 → 128 로 다운샘플
    RandomCrop(128, padding=4), # 살짝 shift 만
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3),
])
transform_val = Compose([
    Resize(128), CenterCrop(128), ToTensor(), Normalize([0.5]*3, [0.5]*3),
])
```

### 5.3 학습 루프 (핵심만)

```python
scaler = GradScaler()
for epoch in range(num_epochs):
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        with autocast():
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
    scheduler.step()
    if epoch % eval_every == 0: evaluate(...)
    save_checkpoint(...)
```

Optimizer = SGD(momentum=0.9, wd=5e-4) 또는 AdamW(wd=0.05). ResNet-18 는 SGD + cosine 이 표준.

### 5.4 Family-aware eval

```python
# modules/family_eval.py
def family_acc(preds, targets, canonical_db_path, classes):
    # preds, targets: list of class_idx
    # classes: list[str] e.g. ["U+4E00", ...]
    db = sqlite3.connect(canonical_db_path)
    exact = 0; partial = 0
    for p, t in zip(preds, targets):
        if p == t: exact += 1; continue
        # 같은 family 조회
        fam_of_t = query_family(db, classes[t])
        if classes[p] in fam_of_t: partial += 1
    return exact / N, partial / N, (exact + 0.5 * partial) / N
```

---

## 6. 하드웨어 / 시간 견적

| 단계 | VRAM | 시간 | 디스크 |
|---|---|---|---|
| S2-A 파일럿 mini | ~2 GB | 10분 | ckpt ~100 MB |
| S2-B 파일럿 500cls | ~4 GB | 30~60분 | ckpt ~300 MB |
| S2-C production 20 epoch | ~6~8 GB (batch 256, 128², AMP) | ~38 h | ckpt 5 × 300 MB + logs |
| Eval 전체 val | ~4 GB | 10분 | metrics.jsonl |
| ONNX + INT8 (선택) | - | 30분 | tflite ~20 MB |

RTX 4080 16 GB 면 batch 512 까지 가능, 그 경우 step 수 절반 → 약 20h. 파일럿 시 step time 측정 후 결정.

---

## 7. 체크리스트

### Midpoint (내일 4/20) 전까지 — 최소
- [x] `train_engine_v1/` 폴더 + skeleton 코드 (modules + scripts + configs) — 2026-04-19
- [x] CPU end-to-end smoke (`00_cpu_smoke.py`) 통과 — train/eval/ckpt reload/curve 저장 전 경로
- [x] `21_eval.py` (ckpt 로딩 + top-k 리포트) CPU 테스트 완료
- [x] `modules/family_eval.py` (canonical_v2 SQLite family_members_json 로 partial-credit) — 103k 패밀리 로드 확인
- [x] `22_export_onnx.py` CPU parity 통과 (torch↔onnx diff < 2e-6, opset 17, dynamo=False)
- [ ] 50~100 클래스 mini 학습 GPU 실행 → loss ↓ curve (production 이미지 생성 끝나면 수행)
- [ ] 슬라이드용 figure 1 장

### Final (5/4) 전까지
- [ ] Stage 1 production 완료 + manifest 전체 확보 (현재 진행 중)
- [ ] font-unseen split 로직 확정 (val 전용 font face 목록)
- [ ] S2-C production 학습 실행 (~2 일)
- [ ] S2-D 평가 5 항목 (top-1/5, family-aware, confusion top-20, [선택] ablation, [선택] tflite)
- [ ] 보고서 + demo 스크립트

### 보류
- [ ] Coral TPU edgetpu_compile + Pi 실측 — 시간 극히 남을 때만
- [ ] Contrastive head + variant graph positive pair — stretch goal

---

## 8. v2 / v3 호환성 주의

- Stage 1 v3 엔진의 모든 augment 는 이미 "real-camera simulation" 범위에 포함 → Stage 2 는 추가 augment 최소화. 이게 지켜지지 않으면 "Stage 1 엔진이 Stage 2 성능 상한을 결정" 이라는 doc 07 § 결론과 어긋남.
- class index 는 **codepoint notation 의 sorted order** — canonical DB v2 의 class_list_v1.jsonl 순서와 동일하게 맞춰두면 추후 DB join 이 깨끗.

---

## 변경 이력

- 2026-04-19 — Track 2 프레이밍 결정 + midpoint 내일 일정 반영 초안. Stage 1 production run 진행 중에 작성.

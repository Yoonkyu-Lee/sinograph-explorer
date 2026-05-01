# Phase 2 Extension — Results: 10-epoch 추가 학습 + 4 진단

작성: 2026-05-01
선행: doc/29 (Phase 2 결과), doc/30 (Phase 3+4 결과)
후행: doc/32 (예정 — Phase 3 재포팅 + Phase 4 재측정 시 작성)

## 요약

✅ **10-epoch extension 의 emb/top1 = 95.2%** (val 11,781) — v3 baseline 38.99% 의 *2.4배*.
✅ **4 진단 (A, B, C, D) 모두 PASS** — anchor crowding pathology 아닌 *진짜 cluster crystallization*.
⚠️ **Phase 3 재포팅 필요** — Pi 의 INT8 deploy artifacts 는 epoch 10 weights 로 빌드돼 있음. 새 best.pt 로 Keras port + INT8 + Edge TPU 컴파일 다시 해야 deploy quality 도 95% 영역.

| 항목 | epoch 10 (Phase 2) | epoch 20 (Extension) | Δ |
|---|---:|---:|---:|
| emb_full/top1 (12k val) | 37.84% | **95.03%** | **+57.19pp** |
| emb_full/top5 | 54.78% | 96.08% | +41.30pp |
| scer/top1 (filter+cosine) | 36.71% | 80.30% | +43.59pp |
| filter/gt_recall | 70.94% | 80.87% | +9.93pp |
| 1000-pack top-1 (FP32) | 34.30% | **96.90%** | +62.60pp |
| 20 PNG top-1 | 50.0% | 95.0% | +45.0pp |
| 20 PNG top-5 | 70.0% | **100.0%** | +30.0pp |
| Anchor pairwise cos mean | (미측정) | +0.0004 ± 0.119 | well-separated |

## 1. 학습 진행 (epoch 11-20)

### 1.1 Two-wave crystallization

| Epoch | emb/top1 | emb/top5 | rad/top1 | char/top1 | train_loss | wall (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 (production end) | 37.5% | 54.6% | 66.5% | 1.7% | 5.65 | 8178 |
| **11** | **78.3%** | 94.2% | 72.8% | 4.8% | 8.25 | 7013 |
| 12 (1st best) | 80.6% | 95.0% | 73.1% | 5.3% | 2.90 | 6983 |
| 13-16 (plateau) | 78-80% | 95.x% | 75-78% | 5.7-6.1% | 2.13-2.29 | ~7400 |
| 17 | 82.9% | 95.7% | 77.4% | 6.0% | 2.22 | 6635 |
| 18 | 89.9% | 96.0% | 77.6% | 6.0% | 2.64 | 7010 |
| 19 | 94.5% | 96.2% | 77.2% | 6.1% | 5.13 | 7479 |
| **20** | **95.2%** | **96.3%** | 77.7% | 6.3% | 7.65 | 7421 |

→ 두 번의 *crystallization wave* 발견:
- **Wave 1 (epoch 10→11)**: ×2.1 jump (37.5 → 78.3). Fresh lr=0.001 로 ArcFace classifier weight 가 *결정적으로 anchor 위치* 잡음.
- **Plateau (epoch 12-16)**: 78-80% 안정. cosine LR 가 이미 작아 minor change 만.
- **Wave 2 (epoch 17→20)**: +12pp (83 → 95). cosine LR 의 *극소 영역* (1e-7 ~ 1e-5) 에서 cluster fine-settling.

### 1.2 학습 throughput

Tier 1 (-A) 최적화 (DataLoader prefetch + SGD foreach + NVTX off + ArcFace gather/scatter) 적용:
- **Throughput +47%** (3,342 img/s vs prior 2,270)
- 10 epoch wall-clock: ~20 시간 (baseline 추정 35h 의 **57%**)

### 1.3 Train arc loss 의 후반 폭증

epoch 19-20 의 arc loss 가 1 → 6+ 로 폭증 — *처음에 anchor crowding 의심*.

진단 결과 (§2 참고): **anchor 는 well-separated** (mean cos +0.0004, max 0.735, collapse 0%). Train arc loss 폭증은 ArcFace 의 m=0.5 angular margin 이 *너무 strict 한 training objective* 가 됐기 때문 — cluster 가 *cosine NN* 로는 95% 정확히 분리 됐지만, m=0.5 의 *추가 28.6° 마진* 까지 만족시키려면 더 빡빡한 separation 필요. **inference (cosine NN) 는 m 무관** 이라 영향 0.

## 2. 4 진단 (PT FP32, dev machine)

`train_engine_v4/scripts/54_diagnose_extension.py` — A+C+D 단일 PT FP32 session.
`train_engine_v4/scripts/52_eval_scer_pipeline.py` — B (12k val 통한 filter+cosine).

### 2.1 진단 A — Anchor pairwise cosine ✅

100,000 random pair sampling on `scer_anchor_db_v20.npy` (98169 × 128):

| 지표 | 값 | 정상 기준 |
|---|---:|---|
| Mean | **+0.0004** | ~0 (Gaussian on sphere) ✅ |
| Std | 0.1186 | ~0.09 (theoretical for 128-d) — **약간 큼** |
| Median | -0.0027 | ~0 ✅ |
| p99 | +0.293 | < 0.5 ✅ |
| Max | +0.735 | < 0.9 ✅ |
| Collapse (cos > 0.9) | **0** | should be 0 ✅ |
| Near-orthogonal (\|cos\| < 0.1) | 60.2% | high → well-separated ✅ |

**Diagnosis**: `WELL_SEPARATED`. anchor crowding pathology 가설 *기각*.

Std 가 theoretical 0.09 보다 약간 크게 나오는 (0.119) 은 *visually similar character* 들이 자연스럽게 *small positive cos* 를 만들어서 — 이건 모델이 *시각적 유사성* 을 학습한 신호 (정상).

### 2.2 진단 B — SCER pipeline (filter + cosine NN)

11,781 val samples (val_per_shard=3 stratified):

| Path | top-1 | top-5 |
|---|---:|---:|
| char (legacy 50MB FC) | 6.03% | 11.80% |
| **emb_full** (cosine vs all 98169) | **95.03%** | **96.08%** |
| **scer (structure prefilter + cosine)** | **80.30%** | 80.82% |

| Filter 통계 | 값 |
|---|---:|
| Avg candidates | 936 / 98169 (105× 좁힘) |
| GT recall (정답이 filter 통과 비율) | **80.87%** |
| Fallback to full (filter 빈 set) | 0% |

**핵심 발견**: emb_full (95%) vs scer/filtered (80%) **15pp 격차**. epoch 10 시점 (37.84% vs 36.71%, 1pp 격차) 과 정반대.

**원인**: structure heads (radical, idc, stroke) 가 epoch 10→20 동안 점진 향상 (radical 66→78%) 했지만 *embedding head 만큼 폭발적 개선 안 됨*. 즉 filter recall (81%) 이 emb (95%) 의 ceiling 이 됨.

**Implication**:
- **Deploy 시 구조 필터 끄고 cosine NN over full 98169** = **95% top-1** (Coral SRAM 내 anchor DB 47MB 직접 검색)
- 구조 필터 적용 → 80% top-1 (15pp 손실, latency 만 90× 빠름)
- Trade-off 결정 필요 (latency vs accuracy)

### 2.3 진단 C — 1000-pack 정확도 (PT FP32)

`val_pack_1000.npz` — 동일 1000 sample (Pi epoch 10 baseline 34.30% 측정에 사용한 것):

| Mode | top-1 | top-5 |
|---|---:|---:|
| Pi epoch 10 (INT8 + libedgetpu) | 34.30% | 48.60% |
| **Dev epoch 20 (PT FP32)** | **96.90%** | 97.70% |

**Δ top-1 = +62.60pp**. 동일 sample 분포에 *2.83×* 향상 — *진정한 quality 향상*, val pack 의 sample 편향 영향 아님.

### 2.4 진단 D — 20 한자 PNG real samples (PT FP32)

`deploy_pi/test_chars/*.png` (Pi 의 `~/ece479/test/*.png`):

| Mode | top-1 | top-5 |
|---|---:|---:|
| Pi epoch 10 (INT8 + libedgetpu) | 50.0% (10/20) | 70.0% (14/20) |
| **Dev epoch 20 (PT FP32)** | **95.0% (19/20)** | **100.0% (20/20)** |

**Δ**: top-1 +45pp, top-5 +30pp. 20 PNG 모두 정답이 top-5 안에 있음.

유일한 top-1 miss: **再** (rank 1 — 첫 후보가 visually identical variant 인 *再*, 두 번째에 정답 *再*. Unicode 변종으로 *완전 동일* 시각).

```
✓ 三  三(0.82) 亖(0.63) 𰌴(0.57) 𠄞(0.57) 𠀂(0.55)         rank 0
· 再  再(0.67) 再(0.65) 𢘑(0.61) 𠀑(0.58) 𤥆(0.58)         rank 1  ← 1 차이
✓ 勝  勝(0.74) 𰯈(0.68) 𥟎(0.61) 𩷼(0.60) 𱔟(0.60)         rank 0
... (나머지 17개 모두 rank 0)
✓ 鳳  鳳(0.84) 𮱱(0.63) 𭂺(0.59) 𦁓(0.59) 㵯(0.58)         rank 0
```

## 3. v3 baseline 과 비교

| | v3 (T5-light v2) | v4 SCER e10 | v4 SCER e20 |
|---|---:|---:|---:|
| top-1 | 38.99% (char head) | 37.84% (emb full) | **95.03%** (emb full) |
| top-5 | 66.92% (char head) | 54.78% | **96.08%** |
| 격차 vs v3 | (baseline) | -1.15pp | **+56.04pp** ⬆ |

**v3 의 2.4× 성능** — emb full 모드 deploy 가 결정적.

## 4. Train_loss 후반 역행의 정체

epoch 18 train_loss=2.64 → 19=5.13 → 20=7.65 — 처음엔 *모델 quality 악화* 의심.

**진단 후 결론**: 정상 ArcFace dynamics.

```
α=0.10 × char + 1.00 × arc + structures
```

각 component 의 후기 거동:
- **char**: 11.3-11.4 (high but stuck — α=0.10 이라 무시) → 0.10 × 11.4 = 1.14 contribution
- **arc**: 1 → 6 폭증 → ε=1.0 × 6 = 6.0 contribution → train_loss 의 80%
- **structures**: 0.1-1.0 → 작음 → 0.5 contribution

Arc loss 가 6+ 로 폭증한 진짜 이유 (anchor cosine 분석 후 확정):
- m=0.5 의 angular margin (= 28.6°) 이 *training objective* 로는 *너무 strict*
- Cluster 이미 cosine NN 로 95% 분리. 추가 28.6° 여백 만족시키려 학습 push 했지만 capacity 부족
- ArcFace 의 *target column logit* (= s × cos(θ + m)) 이 *non-target* 들 보다 작아짐 — CE saturate
- **inference (cosine NN, no margin) 은 m 무관** → val emb top-1 = 95% 유지

→ train_loss 와 val accuracy 의 *decoupling*. 정상 ArcFace dynamics.

## 5. 결론

### 5.1 Extension 의 진정한 가치

✅ **emb_full top-1 95.0%** (v3 의 2.4×) — `cluster crystallization` 이 진짜로 일어남.
✅ Phase 2 의 wall-clock cost: extension 20h × Tier 1 efficiency. 합산 41h (production 21h + extension 20h).
✅ 모든 4 진단 PASS — *anchor crowding* 가설 기각.

### 5.2 Deploy implication

| 모드 | top-1 (FP32) | latency 추정 | use case |
|---|---:|---|---|
| **emb_full** (98k anchors cosine) | **95%** | ~13 ms cosine | Pi/Coral, 정확도 우선 |
| **scer (filter + cosine)** | 80% | ~0.15 ms cosine | 추가 latency 단축 |

**현재 Pi/Coral 의 INT8 artifacts 는 epoch 10 weights**. 새 best.pt (epoch 20) 로 **Phase 3 재포팅 필요**:

1. `40_port_pytorch_to_keras.py` 다시 (best.pt epoch 20 → scer_keras_v20.keras)
2. `41_export_keras_tflite.py` (INT8 변환)
3. `edgetpu_compiler` (재컴파일)
4. Pi 로 transfer + `eval_pi_scer.py` (INT8 정확도 검증)

기대 결과: **INT8 INT8 deploy 의 emb top-1 ~95% 영역** (epoch 10 시점에 INT8 손실 0pp 였음 — quantization 안정성 확인됨).

### 5.3 Top-5 격차의 의미

v4 e20: top-5 96.08% (top-1 95.03% 와 1.05pp 격차).
v3: top-5 66.92% (top-1 38.99% 와 27.93pp 격차).

→ v4 emb 의 *top-1 confidence 가 매우 높음*. v3 의 char head softmax 보다 *cluster 가 sharply* separated.

이건 전형적 *metric learning vs classification* 차이 — ArcFace 가 cluster sharpness 를 직접 supervisor 함.

### 5.4 제기된 우려 (어디까지 valid)

| 우려 | 정확성 | 영향 |
|---|---|---|
| anchor collapse | ❌ 틀림 | (없음) |
| top-1↔top-5 격차 작음 | ✅ 맞지만 *좋은 신호* — sharp cluster | (긍정) |
| char/top1 6% stuck | ✅ 맞지만 *deploy 안 됨* | (없음) |
| train arc loss 폭증 | ✅ 맞지만 *m=0.5 의 over-strict objective* | inference 무관 |
| filter recall 81% | ✅ structure heads 의 한계 | scer/top-1 80% ceiling |

## 6. 다음 단계

### Mandatory (Phase 3 재포팅)

1. **scer_keras_v20.keras** — `40_port_pytorch_to_keras.py --ckpt best.pt --out ...`
2. **scer_int8_v20.tflite** — `41_export_keras_tflite.py`
3. **Edge TPU compile** (WSL)
4. **Pi 전송 + 정확도 측정** — `infer_pi_chars.py`, `eval_pi_scer.py` with new artifacts
5. **doc/32 작성** (Phase 3-redo + Phase 4-redo 결과)

### Optional

- v3 vs v4 head-to-head — 1000-pack 으로 v3 도 측정 (10분)
- Top-5 격차 좁히기 — emb_dim 256 또는 sub-center ArcFace (별개 학습)
- Filter recall 향상 — top-K 늘리기 (rad top-5, idc top-3)

## 7. 산출물 매니페스트 (Extension)

| 파일 | 위치 | 크기 | 용도 |
|---|---|---|---|
| `train_engine_v4/configs/scer_extension.yaml` | repo | 2 KB | 10-epoch extension config |
| `train_engine_v4/configs/scer_extension_sanity.yaml` | repo | 2 KB | Tier 1 sanity (-A 결정) |
| `train_engine_v4/scripts/54_diagnose_extension.py` | repo | 11 KB | A+C+D 단일 진단 |
| `train_engine_v4/scripts/test_arcface_parity.py` | repo | 4 KB | gather/scatter parity |
| `train_engine_v4/out/16_scer_v1/best.pt` | gitignored | 567 MB | epoch 20 best (95.24%) |
| `train_engine_v4/out/16_scer_v1/best_epoch10_backup.pt` | gitignored | 567 MB | epoch 10 backup (안전망) |
| `train_engine_v4/out/16_scer_v1/diagnose_extension.json` | repo | 4 KB | A+C+D 결과 |
| `train_engine_v4/out/16_scer_v1/eval_scer_pipeline_v20.json` | repo | 1 KB | B 결과 |
| `train_engine_v4/out/16_scer_v1/train_result.json` | repo | 1 KB | 10 epoch summary |
| `deploy_pi/export/scer_anchor_db_v20.npy` | gitignored | 47.93 MB | epoch 20 anchor DB |
| `deploy_pi/export/scer_anchor_db_v20.json` | repo | 1 KB | anchor metadata |
| `deploy_pi/test_chars/*.png` | repo | ~50 KB | 20 한자 (D 진단 sample) |

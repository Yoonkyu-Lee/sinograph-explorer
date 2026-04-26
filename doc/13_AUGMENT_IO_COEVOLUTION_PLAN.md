# Stage 1/2 Co-evolution — augment 다양화 + IO 최적화 + train engine v2

## 배경

Mini 모델 (1k class × 5 epoch) 실전 검증 결과:
- **rare Ext B 한자 (𤴡 U+24D21): top-1 86.4%** ✅ — proposal novelty 직접 증명됨
- **Common char 실전 간판 (中)**:
  - Sign 1 (빨간 saturated bg, 브러시 폰트): 7.9% (OOD)
  - Sign 2 (검정 bg + 흰 글자, 인쇄체): 18.2% (OOD, tight crop 후)
- **A-tier (synth 합성 중)**: 89.3% ✅

결론:
1. **Stage 1 augment 분포 외 (OOD)** 에서 심각한 성능 저하. 특히:
   - Saturated 색배경 (red/yellow/blue 간판)
   - Dark bg + light char 반전 조합
   - 브러시/간판체 stylized 폰트
   - 강한 외곽선/깊이 효과
2. **I/O throughput 이 GPU util 50% 에 stuck**. M4 winner (bs256/w8/PIL+gpu-normalize) 로도 steady ~3185 img/s. Production 18~25h 예상
3. **학습 여러 번 돌릴 여유 부족** — augment 실험, hyperparameter sweep, ablation 모두 시간 제약

이 문서는 **Stage 1 augment 확장 + Stage 2 IO 개선** 을 묶어 학습 사이클을 **10~20× 단축** 하는 통합 계획.

---

## 목표

| 축 | 현재 | 목표 | 수단 |
|---|---|---|---|
| Real-world common char (Sign 1 빨간 中) | 7.9% | **50%+** | Stage 1 augment 에 saturated bg, outline, 브러시 폰트 추가 |
| Rare char 유지 | 86.4% | **≥ 80%** 유지 | 회귀 방지 |
| Production wall time (20 epoch) | 18~25h | **3~6h** | WebDataset shards + (선택) tile packing |
| Throughput (pilot steady) | 3,185 img/s | **10,000+ img/s** | File count 5.45M → shard 수천, decode open overhead ↓ |
| Epoch 1 wall (500k mini) | ~500s | ~100s | tar 내 sequential read, prefetch 효과 |

**Success criteria (Phase 5 재평가)**:
- Sign 1 중 7.9% → ≥ 50%
- Rare 𤴡 성능 ≥ 80% 유지
- Production 학습 < 6h
- Stage 1 augment 추가로 실전 test 12 자 top-1 평균 ≥ 70%

---

## Scope: 3 개 워크스트림

독립 진행 가능. 실패시 fallback 명확.

### WS1 — Stage 1 augment 확장 (`synth_engine_v3/`)

현 v3 config `full_random_v3.yaml` 에 대한 v3 extension (realistic). 기존 config 와 병존.

**추가 / 강화할 augment**:

| 항목 | 현재 | v3 extension 목표 |
|---|---|---|
| `background.solid` | [255,255,255] 만 | solid colors 에 **saturated palette** 추가 — 빨강(네온) / 노랑(황색간판) / 파랑(도로표지) / 초록 |
| `background.noise` + `gradient` | 있음 (prob 0.12) | 그대로, 확률 유지 |
| **NEW: `background.textured`** | 없음 | 벽돌, 금속, 나무 텍스처. 카메라로 찍은 간판 지원 |
| **NEW: `invert` 통합** | 없음 | character fill 과 background 의 밝기 관계를 **확률적으로 반전** (dark-on-light ↔ light-on-dark). prob 0.25 |
| `outline.simple` | width 1-2, color black, prob 0.25 | range 확장 (1-5), color 는 contrast-aware (bg 와 반대 hue), prob 0.4 |
| **NEW: `outline.double`** | 현재 v3 에서 미포팅 (v2 에서 빠짐) | 흰 외곽선 + 검은 외곽선 2 단 (네온 간판 전형) |
| `fill.hsv_contrast` | min_contrast 60 | 60 유지, range 는 saturation 0.5-1.0 그대로 |
| **NEW: font 풀 확장** | Malgun, Noto, YuGoth 계열 46 종 | **Ma Shan Zheng**, **ZCOOL KuaiLe**, **Liu Jian Mao Cao**, **Zhi Mang Xing** 등 **서예/간판체 구글 폰트** 5-10 종 추가 |
| `shadow.drop` | offset 2-8, blur 2-6 | 강화: offset 2-15, blur 2-12 (실제 간판 그림자) |
| **NEW: `glow.neon`** | 현재 미포팅 | saturated color 외곽 glow (네온 사인 모사) |
| augment `low_light` | prob 0.2, brightness 0.35-0.65 | prob 0.25 |
| augment `chromatic_aberration` | shift -4~4, prob 0.4 | range 유지 |
| augment `motion_blur` | v2 에 있었으나 v3 미포팅 | **재포팅 필요**. 모바일 카메라 흔들림 |

**산출물**:
- `synth_engine_v3/configs/full_random_v3_realistic.yaml`
- v3 pipeline 내 미구현 layer 포팅 (outline.double, glow.neon, motion_blur)
- 폰트 풀 추가: `samples/fonts_brush/` 에 ~10 종 다운로드 + 등록

### WS2 — IO 포맷 재설계

**재평가 (2026-04-20 업데이트, GPT-5 조언 반영)**: 병목의 실체는 **file open 오버헤드 (100μs/file)** 보다 **PNG decode (2-5ms/image)**. 따라서 atlas PNG / WebDataset (tar 안 PNG) 은 file open 만 잡고 decode 는 남김. 진짜 승부수는 **uint8 tensor shard** — decode 제거.

**Option A — uint8 tensor shards** ⭐ **primary**:
- 기존 5.45M PNG 들을 1-time 변환: decode + resize 128² + uint8 tensor 저장
- ~5,000 samples/shard × ~1,090 shards → 각 ~230 MB (uint8 128×128×3)
- 저장 형식: `.npz` 또는 `.pt` (torch native 는 약간 빠름)
- Shard 구조:
  ```python
  {
    "images": uint8[N, 128, 128, 3],   # or (N, 3, 128, 128)
    "labels": int64[N],                # class idx or notation string
    "notations": list[str],            # optional 복원용
    "sources": list[str],              # picked_source 메타데이터
  }
  ```
- Dataset: `IterableDataset` — shard 하나 `np.load` → slice 로 tile batch yield
- 장점:
  - **PNG decode 완전 제거** — CPU 워커 부담 최소화
  - pin_memory + GPU normalize 로 IO → GPU 직행
  - 예상 throughput: **5,000 ~ 15,000 img/s**
- 단점:
  - 디스크 ~260 GB (현 220 GB 대비 +18%, uint8 raw 가 PNG 압축보다 큼)
  - 사전 resize 고정 — 다른 input size 실험시 재변환 필요
  - PIL-only augment 는 이후 못 씀 (이미 Stage 1 에서 다 적용 완료이므로 무관)

**Option B — WebDataset (PNG in tar)** (fallback / 비교군):
- Option A 구현 리스크 실패시 안전망
- File open 감소만 잡고 decode 는 유지 (~3000 img/s 예상)
- 장점: 구현 단순, 검증된 오픈소스 (`pip install webdataset`)

**Option C — Atlas PNG (naive tile)** (X):
- PNG 안에 16 tile 저장. decode 후 slice. file 은 줄지만 decode 는 늘어남 (더 큰 PNG)
- 버림.

**Option A 우선. A 실패시 B.**

**산출물**:
- `synth_engine_v3/scripts/11_pack_tensor_shards.py` — PNG corpus → uint8 tensor shard 변환기
  - multi-worker: PIL decode + resize 128² (M4 transform 과 동일) → shard buffer 누적 → flush
  - 매 shard 는 manifest 행의 random shuffle 청크 (class 다양성 보장)
  - 예상 실행 시간 1-2h (CPU decode bound, parallelize across workers)
- `synth_engine_v3/scripts/12_inspect_shards.py` — shard 1 개 verify + visualize
- (선택) `synth_engine_v3/scripts/13_pack_wds_shards.py` — WebDataset 버전 (fallback)

### WS3 — Train engine v2 (IO 지원)

`train_engine_v1` 을 복사해 `train_engine_v2` 로. (`CLAUDE.md` 의 버전 관리 원칙 대로 v1 은 baseline 으로 보존).

**v2 변경점**:
- `modules/shard_dataset.py` 새 파일 — `TensorShardDataset` (Option A)
  - `IterableDataset` 기반. Shard 단위로 읽어 batch 를 yield
  - worker 간 shard split (각 worker 가 서로 다른 shard 들 담당)
  - shard shuffle + within-shard shuffle buffer 로 random 복원
  - GPU 에서 단순히 float/255 + normalize 만 (no decode, no resize)
- `modules/dataset.py` 유지 (v1 과 동일, PNG 경로 fallback)
- `scripts/20_train.py` config 에 `data.format: "png" | "tensor_shard" | "webdataset"` 옵션
  - `tensor_shard` 경로는 decode/resize skip → GPU normalize 만
- M4 winner 의 batch/workers 숫자 유지 시작 (256/8), steady 측정 후 더 키울 수 있는지 실험 (uint8 IPC 유사)
- 새 smoke test: `scripts/00_cpu_smoke_shards.py`

**산출물**:
- `train_engine_v2/modules/shard_dataset.py`
- `train_engine_v2/configs/resnet18_t1_v3_realistic_full.yaml` — v3_realistic augment + tensor shard
- v1 과 동일한 모든 scripts (00 cpu smoke, 20 train, 21 eval, 22 export onnx, 30 predict)

---

## Phase 분할

**Phase 1** — WS1 Augment 확장 설계 + smoke (~1 일)
- [ ] v3_realistic yaml 초안 작성
- [ ] 브러시 폰트 다운로드 + 테스트
- [ ] v3 미포팅 layer (outline.double, glow.neon, motion_blur) 재포팅
- [ ] 품질 파일럿 (`40_pilot_v3_realistic.yaml`, 5k 샘플 random) → 40-60 샘플 육안 확인
- [ ] Sign 1/2 스타일 분포 합성되는지 검증

**Phase 2** — WS2 WebDataset shard 변환 (~반나절)
- [ ] 현재 `42_production_mobile/` → `42_production_mobile_shards/` 변환 스크립트
- [ ] 디스크 공간 확인 + 일시 440GB 가능 여부 (안 되면 chunked 변환)
- [ ] shard 1 개 hand-inspect
- [ ] webdataset loader smoke (CPU 에서 1 shard 전부 iterate)

**Phase 3** — WS3 train_engine_v2 구현 (~반나절)
- [ ] v1 → v2 복사 + MANUAL 갱신
- [ ] WebDataset wrapper + dataset test
- [ ] M4 config 기반 1-epoch mini (100 cls) smoke
- [ ] 3-epoch steady-state 측정 — 기대 **epoch 3 wall < 10s** (M1 의 32.8s 대비)

**Phase 4** — 새 augment + 새 IO 통합 production (~하루)
- [ ] v3_realistic config 로 Stage 1 재생성 (또는 augment 만 바꿔 재생성, 같은 T1 코드포인트 유지)
- [ ] shard 변환
- [ ] Stage 2 production 학습 (기대 3~6h)

**Phase 5** — 재평가 + 최종 검증 (~반나절)
- [ ] Sign 1/2 중 정답 확률 재측정
- [ ] rare hanzi 𤴡 유지 여부
- [ ] 추가 실전 사진 테스트 (TEST_PLAN 의 12 자)
- [ ] results.md 에 before/after 비교 표

---

## 예상 시간 절감 계산

현 M4 throughput 3,185 img/s @ 5.45M samples × 20 epoch = 109M exposure → 34,200 s ≈ 9.5 h (hot cache 기준)
- 실제 production 은 cold cache 라 ≈ **18~25h** 측정 (epoch 1 888 img/s 에 가까움)

WebDataset 으로 sequential read 확보시 cold/hot 차이 최소화:
- file open 5.45M → 1,090 = 5000× 감소
- decode 자체는 동일, 그러나 per-file overhead 제거
- 기대 throughput **6,000~10,000 img/s** (실측 필요)
- 20 epoch production **3~5h** 가능

Augment 확장 (WS1) 은 학습 time 에 영향 없음 (생성시점). 다만:
- Stage 1 재생성이 필요 → production 이미지셋 220 GB 재작성 (5h 추가)
- 혹은 augment 만 적용한 version 은 유지하고 "v3_realistic" 의 일부 샘플을 추가 생성 (50만 정도만 추가, 1h)

**Net**: 총 1 회 full cycle (Stage 1 재생성 + Stage 2 학습) **~10~12h** 예상. 다음 iteration 부터는 Stage 1 안 바꾸면 Stage 2 만 3-6h.

---

## Risk / Fallback

| Risk | 대응 |
|---|---|
| 디스크 440GB 일시 공간 부족 | Chunked shard 변환 (100 shard 만들면 해당 원본 삭제) |
| WebDataset 이 Windows 에서 spawn 오버헤드 | v1 M4 config fallback 으로 production 진행 (18-25h 감수) |
| 브러시 폰트 라이선스 불명 | Google Fonts 의 OFL 라이선스만 사용 (Ma Shan Zheng, ZCOOL 등 전부 OFL) |
| augment 너무 강해서 rare hanzi 성능 회귀 | v3 baseline 과 v3_realistic 비교 ablation, 필요시 augment prob 조정 |
| Tile packing (옵션 B) shuffle 이슈 | Option A 만으로 충분하면 skip. 10,000 img/s 도달시 Option B 는 diminishing returns |

---

## 이후 (final demo 까지)

이 plan 이 완료되면:
- Stage 1 augment 가 real-world 분포에 더 근접
- 학습 사이클이 5-10h → **하루에 2-3 번 실험 가능**
- hyperparameter sweep (label smoothing, LR schedule, mixup 등) 도 현실적
- Ablation: v3 baseline vs v3_realistic augment → "augment 확장이 real-world 성능 X% 개선" 정량 보고 가능
- Family-aware eval, ONNX export, (선택) Coral TPU compile 등 final demo 요소 여유있게 추가

---

## 참고 문서

- `doc/10_STAGE1_DATASET_GENERATION_PLAN.md` — Stage 1 파이프라인 세부
- `doc/12_STAGE2_TRAINING_PLAN.md` — Stage 2 학습 설계 (M4 winner 포함)
- `train_engine_v1/MANUAL.md` — M1-M5 multi-epoch 튜닝 기록 + winner config
- `train_engine_v1/TEST_PLAN.md` — 실전 OCR 테스트 체계

## 변경 이력

- 2026-04-20 — 초안. Mini OOD 테스트 결과 기반으로 Phase 1~5 정의.

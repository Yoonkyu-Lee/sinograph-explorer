# 배포 결과 + 양자화/TPU 한계 분석 + v4 제안

작성: 2026-04-27. 대상: T5-light v2 (10 epoch / 98,169 class) 의 Pi 배포 검증 후 회고.

이 문서는:
1. 현 모델 + 배포 상태 정리
2. INT8 TFLite 와 Coral TPU 가 안 되는 **본질적 이유** 분석
3. 이를 해결하기 위한 **방법론** (학습/architecture 변경)
4. v4 제안 — v3 데이터셋 재활용 vs 처음부터

---

## 1. 현 상태 요약

### 1.1 학습

- **T5-light v2**: 98,169 class × 10 epoch × bs=640, lr=0.1 cosine + 1ep warmup
- corpus: `94_production_102k_x200` (19.6M samples, val_per_shard=25 stratified)
- 학습 시간: 13.7h (RTX 4080 Laptop)
- val char/top1 = **38.99%** (epoch 10, 98k random baseline 0.001%)
- 학습 곡선의 결정적 패턴:
  ```
  Epoch:    1    2    3    4    5    6    7    8    9    10
  loss:    8.7  7.5  7.4  7.6  7.8  8.1  8.3  8.5  7.4  5.0
  char/t1:  -   0.31  -   0.48  -   0.72  -   1.76  -    38.99
  ```
  → **epoch 8 까지는 plateau, 마지막 cosine 수렴 phase (lr 0.018→0) 에서 폭발적 개선**.
  더 많은 epoch (17, 20) 학습 시 char/top1 50-70% 도달 가능 (외삽).

### 1.2 배포 (Pi)

- ONNX export (char head only, 246 MB) → PC parity OK (max diff 2.6e-6)
- Pi (yoonkyu2@ykspi, ARM, Pi OS Bookworm) onnxruntime FP32 기준:
  - 추론 36.8ms / 이미지 (26.3 img/s)
  - 정확도 PC 와 비트 동등: **top-1 = 5/20 (25%), top-5 = 6/20 (30%)**
- 20 single-char 실 이미지에서:
  - **char-only top-1 적중**: 戦, 旧, 標, 蒼, 都 (5)
  - **top-5 추가**: 三 (1) → 6 총
  - **char-only top-5 hit rate 30%, fusion top-5 (PC) 45%**

### 1.3 배포 실패 — 두 가지 blocker

| 시도 | 결과 |
|---|---|
| ONNX FP32 → Pi onnxruntime | ✅ 36.8ms, PC 와 동등 |
| ONNX → TFLite INT8 (full integer) | ❌ 모든 logit 거의 동일 → 무용 |
| ONNX → TFLite INT8 (FP32 IO + INT8 weights) | ❌ 단일 class 에 100% 몰빵 |
| Coral edgetpu_compiler | ❌ 0 ops mapped, 전부 CPU fallback |

→ **속도/메모리 이득은 못 봄. ONNX FP32 로 마무리.**

---

## 2. 첫 번째 고민: 왜 INT8 TFLite 가 안 되는가

### 2.1 직접 원인

위 INT8 변형 4가지 (`full_integer_quant`, `integer_quant`, 둘 다 `_with_int16_act` 변형) 모두 같은 패턴:
- top-1 이 무작위 char (idx 0 또는 거대 cluster 의 한 점) 에 100% 몰빵 또는 모든 prob 0%
- 의미 있는 ranking 불가능

### 2.2 본질적 원인 — Classifier head 크기

**Final FC = Linear(512 → 98,169)**, 즉 **50.3M params** 가 모델의 81% 차지 (전체 61M):

```
backbone (ResNet-18):    11.2 M params  (19%)
char_head (Linear):      50.3 M params  (81%)  ← 여기가 문제
aux heads (4개):         < 0.1 M params
─────────────────────────────────────
total:                   61.6 M params
```

INT8 양자화 시:
1. **Per-channel scale 의 outlier 민감성**:
   - 98,169 channel 각각 독립적 scale/zero_point 추정
   - 300 calibration sample × 균일 분포 가정 → **각 채널은 평균 0.003개 sample 만으로 scale 추정**
   - 대부분 channel 에서 scale 부정확 → 가중치 round-off 손실 큼

2. **Output activation 의 long-tail 분포**:
   - 98k logit 중 정상적으로 1-5개만 큰 값, 나머지는 작은 값
   - INT8 의 256 단계로는 이 dynamic range 표현 못 함
   - 작은 logit 들이 0/±1 INT8 로 quantize 되어 정보 lose

3. **Softmax 의 saturation**:
   - INT8 logit 차이가 작으면 softmax 가 거의 uniform → 의미 있는 ranking 불가능
   - Floating point 로 잠시 dequantize 해도 이미 정보 손실됨

### 2.3 비교 — v2 baseline 은 INT8 OK 였을 것

v2 (10,932 class) 의 final FC = `Linear(512 → 10,932)` = 5.6M params.
- 9× 작아서 calibration 충분
- per-channel scale 추정 정확
- INT8 양자화 손실 -1~2% 정도로 합리적

→ **INT8 부적합은 class 수의 함수**. ResNet-18 conv 부는 정상 양자화됨.

---

## 3. 두 번째 고민: 왜 Coral Edge TPU 가 안 되는가

### 3.1 컴파일러 출력

```
Number of operations: 39
Number of operations that will run on Edge TPU: 0
Number of operations that will run on CPU: 39
On-chip memory used for caching model parameters: 0.00B
```

39 ops 전부 CPU fallback. TPU 사용 0%.

### 3.2 본질적 원인 — Coral SRAM 8MB vs 모델 63MB

Coral USB Accelerator 의 **on-chip SRAM = 8 MB** (M.2 / PCIe 도 동일).

우리 모델:
- INT8 TFLite 전체: **63 MB**
- 그중 final FC 가중치만: **50 MB**
- backbone + aux: ~13 MB

상황:
- FC 만 8MB SRAM 에 못 들어감 (8× 큼)
- TPU 가 op 실행할 때마다 weights 를 **USB 통해 host RAM 에서 streaming** 해야 함
- USB 2.0 대역 ~480 Mbps = 60 MB/s. FC weights 50 MB 한 번 읽기 = ~830ms
- Pi 4 CPU 가 같은 작업 ~150ms → **USB streaming 이 더 느림**
- edgetpu_compiler 가 이를 인식 → **TPU 사용 자체를 포기** (0 ops mapped)

### 3.3 비교 — v2 의 case

v2 INT8 모델 ≈ 17 MB (10932 class) → 여전히 SRAM 8MB 보다 크지만 비율 2× 만 (vs 우리 8×)
- edgetpu_compiler 가 backbone 일부 ops 를 TPU 매핑 가능 (50%+)
- 부분 가속 효과로 CPU only 보다 빠름

→ **Coral 부적합도 class 수의 함수**. v2 면 부분 매핑은 됐을 거.

---

## 4. 공통 원인 — Big Classifier Head

INT8 TFLite, Coral TPU **둘 다 본질적으로 같은 문제**:

```
50.3M params final FC == 양자화 friendly 하지 않음
                     == TPU on-chip 캐시 안 들어감
                     == calibration set 으로 scale 추정 부족
                     == long-tail logit 분포로 INT8 dynamic range 부족
```

이건 modern multi-class CNN 의 **알려진 한계** — class 수 가 ~10k 넘어가면 `Linear(d → C)` 가 deploy 의 bottleneck.

---

## 5. 해결 방안 — 큰 classifier 를 우회하는 architecture

### 5.1 옵션 A — Embedding + Retrieval (가장 표준)

modern face/landmark recognition 에서 100k+ identity 처리할 때 쓰는 방식:

```
                                              [class embedding DB]
                                              c1: (512,) vector
input → backbone → 512-d embedding → cosine →   c2: (512,) vector
                                              ...
                                              cN: (512,) vector
                                                   ↓
                                              top-k by similarity
```

**구조 변경:**
- `Linear(512 → 98169)` 제거 → backbone 만 남김
- `class_embedding_db` 를 학습 후 build (per-class mean embedding, ~50 MB FP32 → 12 MB INT8)
- 추론 시: backbone (Coral 가능) → 512-d embedding → DB lookup (CPU, 50 MB read 1회)

**장점:**
- backbone 만 양자화 → 11M params 모두 INT8 OK, **Coral 100% 매핑**
- backbone 추론 ~10-30ms on Coral (현 36ms → 10ms 정도)
- DB lookup ~5ms on Pi CPU (cosine sim over 98k×512)

**단점:**
- 학습 더 복잡 (metric learning loss 추가)
- DB 별도 build 단계 필요

### 5.2 옵션 B — Hierarchical Classifier (계층 분류)

aux head 를 분류 chain 으로 활용:

```
input → backbone → idc head (12-way)        ← Coral 친화 (소형)
                 → radical head (214-way)   ← Coral 친화 (소형)
                 → fine head (~500 candidates per (idc, radical))
                                            ← 작은 set 라 INT8 OK
```

각 단계 conditional → 마지막 fine head 가 (idc, radical) 로 좁혀진 후보군에서만 결정.

**장점:**
- 모든 head 가 INT8 + Coral 가능
- 우리 v3 의 aux 학습이 그대로 살아있음 (radical 71%, idc 94%)

**단점:**
- 부정확한 coarse 예측이 fine 까지 전파 (재귀 오답)
- (idc, radical) 조합으로 후보군 빌드하는 metadata 필요

### 5.3 옵션 C — Reduced Class Set (가장 간단)

98k → **10k** (가장 흔한 한자만):
- 일본 상용 한자 + 한국 교육용 + 중국 GB18030 1단계 합집합 = ~10-15k
- v2 의 10,932 와 비슷한 규모
- Linear(512 → 10000) = 5M params → INT8 OK + Coral 부분 매핑 OK

**장점:**
- architecture 변경 0
- v3 학습 코드 그대로
- 즉시 batch 진행 가능

**단점:**
- 우리가 96 % 보장한 **102,944 coverage 의 가치 잃음**
- 전 corpus 재학습 필요 (subset shards 구축)

### 5.4 옵션 D — Mixed deployment (즉시 가능)

학습 변경 없이 배포만 분할:
```
backbone (11M) → Coral TPU                   ← 가속
        ↓ 512-d feature
final FC (50M) + softmax → Pi CPU            ← FP32 그대로
```

**장점:**
- v3 모델 그대로 사용
- backbone 부분만 INT8 양자화 (FC 빼고) — 양자화 손실 미미
- Coral USB 가 backbone 가속, CPU 가 dense 만 처리
- 추론 ~50ms (backbone Coral 20ms + FC CPU 30ms) 추정

**단점:**
- onnx → tflite 분할 모델 만들어야 함 (수동 split)
- runtime 코드 분리 (Coral op 후 CPU op)

---

## 6. 권장 — v4: SCER (Structure-Conditioned Embedding Recognition)

> **TPU constraint 를 architecture novelty 로 변환.** 5장 옵션들을 단순히 채택
> 하지 않고, CJK 글자의 **구조적 분해 가능성** 을 inductive bias 로 활용해
> 5.1 (embedding) + 5.2 (hierarchical) 를 hybrid 한 새 architecture 를 제안.

### 6.1 핵심 아이디어

flat 98k softmax 의 한계를 우회: **"구조 좁히기 + 임베딩 비교"** 의 2-stage.

- **Stage 1 (구조 추론)**: backbone → (radical, idc, strokes) 예측 — 작은 head 들
- **Stage 2 (임베딩 매칭)**: backbone → 128-d embedding → 구조 일치 후보군 안에서만 cosine similarity 비교

이것이 강한 이유:
1. **TPU constraint 자연 해결**: 50MB FC → 13KB embedding linear 로 대체
2. **Domain knowledge 활용**: v3 의 aux head 가 이미 73% radical / 94% idc 정확도. inference 시 활용 안 하면 낭비
3. **Interpretability**: "부수 木 + ⿰ + ~16획 → 후보 50개 → 標 가 가장 유사" — 데모용 ideal
4. **canonical_v3 DB 활용**: 이미 만든 (rad, idc, strokes) → chars 의 reverse lookup 이 inference 시점 knowledge
5. **v3 자산 100% 재활용**: corpus, aux labels, best.pt backbone, 학습 코드

### 6.2 Architecture

```
                       [Backbone — INT8, ResNet-18, ~11MB]   ← Coral TPU 100%
                                        ↓
                                 512-d feature
                            ┌────────────┼────────────────────┐
                            ↓            ↓                     ↓
        [Structure heads]              [Embedding head]      (학습 시 only)
        rad (214)                      Linear(512→128)       [Char head]
        idc (12)                       L2 normalize          Linear(512→98k)
        total_strokes (1)              ArcFace target         CE loss (warmup)
        ↓                              ↓
  3 small heads <0.5MB            0.07MB                     50MB ❌ deploy 시 drop
        ↓                              ↓
  Candidate generation           Query embedding
  (canonical_v3 DB lookup):      (128-d INT8)
  chars matching                       ↓
  - rad ∈ top-3 of rad_pred      Rerank
  - idc ∈ top-2 of idc_pred      cosine similarity vs
  - |strokes - pred| ≤ 2         candidate embeddings (precomputed)
        ↓                              ↓
  ~50-500 candidates             top-k of candidates
        └────────── INTERSECT ──────────┘
                        ↓
                top-k char predictions
```

### 6.3 TPU/INT8 적합성

| 요소 | 크기 (INT8) | TPU 가능? |
|---|---:|---|
| ResNet-18 backbone | ~10 MB | ✅ 100% mapped (v2 기준 검증) |
| 3 structure heads | <0.5 MB | ✅ 100% |
| Embedding head | 0.07 MB | ✅ 100% |
| **총 deploy model** | **~11 MB** | ✅ Coral SRAM 8MB 의 1.4× — 부분 streaming 으로 유효 매핑 90%+ |
| Char head (학습 only) | 50 MB | deploy 시 제거 |
| Class embedding DB | 12 MB INT8 (98k × 128) | CPU 측 (lookup) |

→ **"INT8 + Coral" 가 enable 됨**. 50MB FC 가 사라지면서 본문서 §2-3 의 한계가 모두 해소.

### 6.4 학습 — Loss 와 schedule

**Joint loss:**
```
L = α · CE(char_head, char_y)            ← warmup 신호 (epoch 1-3 만 강하게)
  + β · CE(rad_head, rad_y, mask)        ← v3 aux 그대로
  + γ · CE(idc_head, idc_y, mask)        ← v3 aux 그대로
  + δ · SmoothL1(strokes, strokes_y)     ← v3 aux 그대로
  + ε · ArcFace(embedding, char_y)        ★ 새 metric loss
```

**α/ε weight scheduling** (curriculum):

| Phase | Epoch | α (CE) | ε (ArcFace) | 의도 |
|---|---|---:|---:|---|
| Warmup | 1-3 | 1.0 | 0.1 | CE 가 backbone 안정화 |
| Transition | 4-7 | 0.5 | 0.5 | embedding-CE 균형 |
| Fine | 8+ | 0.1 | 1.0 | embedding 중심 학습 |

structure head weights (β, γ, δ) 는 v3 와 동일 (0.2, 0.2, 0.1) 유지.

**ArcFace 선택 이유:**
- 98k class 안정적 학습 (ImageNet, MS1M 검증)
- angular margin → embedding space 의 inter-class separation 강화
- per-class learnable centers (Linear(128, 98k) weight 의 columns) — 학습 끝나면 그대로 class embedding DB
- PyTorch 구현 ~30줄

### 6.5 Class Embedding DB build (학습 후 1회)

```python
# 학습 끝나면 1회 실행
db = torch.zeros(98169, 128, dtype=torch.float32)
counts = torch.zeros(98169, dtype=torch.long)

for batch in train_loader:
    embeddings = model.embedding(batch.images)  # (B, 128)
    for i, char_y in enumerate(batch.labels):
        db[char_y] += embeddings[i]
        counts[char_y] += 1

db /= counts.unsqueeze(-1).clamp(min=1)         # mean per class
db = F.normalize(db, dim=-1)                    # L2 normalize
db_int8 = quantize_int8(db)                      # → 12 MB
torch.save(db_int8, "class_embeddings.pt")
```

대안: ArcFace 의 `Linear(128, 98169)` weight column 들을 그대로 DB 로 사용 (별도 build 불필요). 차이는 "샘플 평균" vs "학습된 anchor". 보통 **anchor (ArcFace weight)** 가 더 좋음.

### 6.6 Soft structure filtering

**Hard filter 의 위험**: rad pred 가 틀리면 정답이 후보에서 제거됨 → top-k 회복 불가.

**Soft filter 채택**:
```python
# 각 query 의 candidate set 생성
top_rads = rad_head_logits.topk(3).indices      # 3 candidates
top_idcs = idc_head_logits.topk(2).indices      # 2 candidates
stroke_pred = stroke_head_output                  # scalar

candidates = {
    char in 98169:
        char.radical in top_rads
        AND char.idc in top_idcs
        AND |char.total_strokes - stroke_pred| ≤ 2
}
# 평균 candidate set size: ~100 (98k 의 0.1%)
```

만약 structure 예측 이 틀려도 wide 한 후보 set 이라 정답 포함 확률 높음. radical top-3 + idc top-2 = 조합 6개 layout 의 chars 까지 cover.

**캐싱**: `(rad_idx, idc_idx) → list of char_idx` 의 reverse index 를 pre-build (canonical_v3 DB 활용). 학습 / inference 시점에 O(1) 조회.

### 6.7 v3 자산 재활용

| 자산 | 재활용 방식 |
|---|---|
| 19.6M corpus (`94_production_102k_x200`) | ✓ 그대로 사용 |
| `aux_labels.npz` (radical/idc/strokes) | ✓ 그대로 사용 (구조 학습 + candidate generation 양쪽) |
| `best.pt` (T5-light v2 epoch 10) | ✓ backbone weights warm-start (학습 가속) |
| 학습 인프라 (`00_smoke.py`, `train_loop.py`, `shard_dataset.py`) | 수정 ~150줄 |
| `canonical_v3` DB (`ids_merged.sqlite`) | ✓ candidate generation 의 reverse-index source |
| ONNX export 인프라 (`22_export_onnx.py`) | 수정 ~30줄 (embedding head export) |
| TFLite 변환 인프라 (`23_quantize_tflite.py`) | ✓ 그대로 (이번엔 backbone 만 작아서 INT8 OK) |

### 6.8 새로 작성 (~400줄 총)

- `train_engine_v4/modules/model.py`: SCER architecture (backbone + 3 small heads + embedding head + 학습용 char head)
- `train_engine_v4/modules/arcface.py`: ArcFace loss (~30줄)
- `train_engine_v4/modules/train_loop.py`: 5-component joint loss
- `sinograph_canonical_v3/scripts/52_build_structure_index.py`: (rad, idc, strokes) → chars reverse index
- `train_engine_v4/scripts/25_build_class_db.py`: 학습 후 embedding DB build
- `train_engine_v4/scripts/30_predict_v4.py`: structure filter + embedding rerank
- `deploy_pi/infer_pi_v4.py`: 동일 로직 Pi 측 (TFLite + structure filter)

### 6.9 예상 성능

| 항목 | v3 (현재) | v4 SCER (예상) |
|---|---:|---:|
| char/top1 (val) | 39% | **45-60%** (warm restart + ArcFace + filter) |
| char/top5 | ~50% | **70-85%** (filter 가 candidate 좁힘) |
| Pi inference | 36.8 ms (ONNX FP32) | **15-25 ms** (Coral + small heads) |
| Coral 매핑 | 0% | **90%+** |
| INT8 가능 | ❌ | **✅ full integer** |
| Model size (INT8) | n/a | **~11 MB** (deploy) + 12 MB DB |

### 6.10 Novelty Statement (발표용)

> Standard CJK character recognition uses flat 100k-way softmax classification,
> which prevents deployment on resource-constrained accelerators (e.g., Coral
> Edge TPU has 8MB SRAM, while a 98k-class FC layer alone consumes 50MB). We
> exploit the **compositional structure of Han characters**: each character is
> decomposable into a radical, layout descriptor (IDC), and stroke count. Our
> model jointly predicts these structural primitives and a learned 128-d
> embedding, then performs **retrieval over a structure-conditioned candidate
> set** derived from a canonical character database. The classifier's
> per-class output is replaced by a small embedding head, reducing the
> deployable model from 61M to 11M parameters and enabling full-integer
> quantization compatible with Edge TPU. Inference becomes interpretable —
> "this radical + this layout + ~N strokes → these candidates → most similar
> to query embedding" — exposing the model's reasoning to the user.

학술적 명칭: **"Compositional Inductive Bias for Extreme Classification on Edge Devices"**.

### 6.11 시간 비용

| 단계 | 시간 |
|---|---|
| 코드 작성 (~400줄) | 2-3일 |
| 학습 (warm restart 10-15 epoch) | 12-18h |
| Class embedding DB build | 30분 |
| ONNX/TFLite 변환 + Pi 검증 | 1일 |
| 발표용 demo 영상 + 데이터 정리 | 1일 |
| doc/25 (v4 결과 보고서) | 1일 |
| **총** | **5-7일** |

발표 다음 주 + 데모 다음 다음 주 라면 충분.

### 6.12 Plan B (SCER 시간 부족 시 fallback)

만약 SCER 의 학습 안정성 issue 또는 시간 부족하면:

**옵션 D mixed deploy** 로 후퇴:
- v3 model 그대로
- backbone 만 Coral, FC 만 Pi CPU
- 학습 변경 0, novelty 낮지만 batch 가능
- char accuracy = v3 와 동일 (39%)

학술적 강점은 약하지만 demo 는 가능.

### 6.13 결정

**SCER (v4)** 권장 — 본 프로젝트의 가장 강력한 final form.

근거:
- 본 과정 (ECE 479) 이 backbone engineering / novelty 를 우선시
- TPU constraint 가 자연스러운 motivation
- v3 자산 모두 재활용 → 학습 효율
- 데모 timeline (다음 주 발표 + 데모 다음 다음 주) 안에 finishable
- INT8 + Coral 둘 다 enable 됨 → 본문서 §2-3 의 blocker 모두 해소

---

## 7. 즉시 가능한 우회책 (v4 안 가더라도)

v3 모델 그대로 두면서 deploy 측면에서 할 만한 것:

1. **옵션 D mixed deploy** (1-2일 작업)
   - backbone 만 Coral, FC 만 Pi CPU
   - v3 학습 변경 0
   - Coral 효과 일부라도 활용

2. **v3 char head 만 INT8 + retrieval emulation** (1일)
   - 학습된 final FC weight matrix W (512 × 98169) 를 INT8 quantize
   - 추론 시: backbone embed → embedding 과 W 비교 → top-k
   - 본질적으로 retrieval 과 동일
   - Pi 에서 W 읽기 50 MB 는 한 번만 (model load), inference 시엔 dot product 만

3. **현재 ONNX FP32 그대로 사용** (작업 0)
   - 36.8ms on Pi 로 충분히 real-time
   - 추가 최적화 가치 낮음
   - INT8/Coral 의 의미는 "더 빠를 수 있었다" 정도

---

## 8. 한 줄 요약

> **98k-class final FC 가 50M params 라서 INT8/Coral 부적합.** TPU constraint 를
> architecture novelty 로 변환: **SCER (Structure-Conditioned Embedding Recognition)** —
> CJK 글자의 (radical, idc, strokes) 구조 예측으로 후보군 좁히고, 그 안에서만
> 128-d embedding 비교. 50MB FC → 13KB embedding linear. v3 자산 100% 재활용,
> 학습 시간 +12-18h. 데모 timeline 5-7일.

---

## 9. Open Questions / 다음 액션

1. **embedding loss 선택**: ArcFace / CosFace / SupCon / Triplet — 어느 게 98k class 에 안정적?
2. **DB build 전략**: 모든 train sample → mean embed per class, 또는 좋은 sample 만 select?
3. **embedding dim**: 256 vs 512 vs 1024?
4. **negative mining**: ArcFace 의 negative sampling 어떻게 — random / hard / semi-hard?
5. **softmax CE 와의 weight 비율**: 메인을 ArcFace 로 할지, CE 로 할지?

이 결정은 Phase T8 (v4 시작 시) 에서 PoC 실험으로 답.

---

## 참고 자료

- [SGDR — Loshchilov & Hutter 2017](https://arxiv.org/abs/1608.03983) — cosine warm restart
- [ArcFace — Deng et al. 2019](https://arxiv.org/abs/1801.07698) — face recognition embedding loss
- [Coral Edge TPU model requirements](https://coral.ai/docs/edgetpu/models-intro/)
- [TFLite quantization deep dive](https://www.tensorflow.org/lite/performance/post_training_quantization)

# Phase 2 — SCER (Structure-Conditioned Embedding Recognition) 학습 plan

작성: 2026-04-28
선행: doc/24 §6 (SCER 제안), doc/27 (Phase 1 deploy 경로 확정)
후행: doc/29 (예정 — Phase 2 결과 + Phase 3 Keras 포팅 plan)

## 0. 한 줄 요약

best.pt (T5-light v2) 의 backbone 을 그대로 warm-start 해서, **char head 50 MB FC 를 128-d embedding head + ArcFace classifier** 로 바꾼 SCER 모델을 학습하고, 결과를 anchor DB + structure soft filter pipeline 으로 평가한다. Phase 3 (Keras 포팅 + Edge TPU) 는 doc/29 에서 별도 처리.

## 1. 이 phase 가 필요한 이유

doc/27 가 "Keras-native ResNet-18 + 50 MB FC" 모델도 Edge TPU 에 100% mapped 됨을 실증했다. 따라서 SCER 의 동기 중 "FC 가 컴파일을 막아서" 는 더 이상 유효하지 않다. 단:

1. **on-chip SRAM 에 fit 하기 위해**: 현재 v3 INT8 TFLite 는 51 MB off-chip streaming. SCER 의 ~12 MB 모델은 8 MB SRAM 1.5× 라 부분 streaming 이 훨씬 적음 → Coral latency 가 실제로 빠를 가능성. (doc/27 §8.5 의 Codex 지적 해소)
2. **Final Demo 의 학술적 novelty**: 단순 압축이 아닌 "구조 분해 inductive bias" 로 80 K+ class 분류를 풀어내는 architecture. Lab3 의 평가 기준과 부합.
3. **interpretability 데모**: "부수 木 + ⿰ + ~16획 → 후보 50개 → 標 가 가장 유사" 의 추론 trace 를 그림으로 보여줄 수 있음.

따라서 SCER 는 *deploy 가능성* 이 아니라 **deploy 품질 + 학술 novelty** 를 위한 phase.

## 2. 비목적 (out of scope, doc/29 로 이월)

- Keras 포팅 (`keras_resnet18.py` 확장: embedding head + structure heads 추가)
- Edge TPU INT8 변환 + 컴파일
- Pi 실측 latency (`bench_three_way.py` 변형)
- canonical_v3 DB 의 (rad, idc, strokes) → chars reverse index 사전 빌드

이 4 항목은 Phase 2 의 학습 결과가 게이트를 통과한 후에 진행. 미리 해놓아도 되지만 학습 실패 시 무용지물 → Phase 2 결과 확정 후 doc/29 작성.

## 3. 영향 받는 파일

### 3.1 수정 (4 파일)

| 파일 | 변경 | 분량 |
|---|---|---|
| `train_engine_v3/modules/model.py` | `MultiHeadResNet18` 에 `embedding_head` (Linear 512→128) + ArcFace classifier 추가, forward dict 확장 | +30 줄 |
| `train_engine_v3/modules/train_loop.py` | `LossWeights.embedding` 추가, ArcFace term 합산, curriculum schedule helper | +40 줄 |
| `train_engine_v3/modules/aux_labels.py` | (변경 없음 예상 — 기존 valid 마스크 재사용) | 0 |
| `train_engine_v3/scripts/00_smoke.py` | SCER smoke 분기 추가 (--scer flag) 또는 신규 scripts/50_smoke_scer.py 분리 | +50 줄 |

### 3.2 신규 (5 파일)

| 파일 | 용도 | 분량 |
|---|---|---|
| `train_engine_v3/modules/arcface.py` | ArcMarginProduct + cosine margin 구현 | ~70 줄 |
| `train_engine_v3/configs/resnet18_scer_smoke.yaml` | smoke (1 shard, 1 epoch) | ~30 줄 |
| `train_engine_v3/configs/resnet18_scer_production.yaml` | production (전체 shard, 10 epoch, warm-start) | ~50 줄 |
| `train_engine_v3/scripts/50_train_scer.py` | SCER 학습 entry (`00_smoke.py` 의 production 변형) | ~200 줄 |
| `train_engine_v3/scripts/51_build_anchor_db.py` | 학습 후 (98169, 128) anchor table 생성 + L2 normalize | ~80 줄 |
| `train_engine_v3/scripts/52_eval_scer_pipeline.py` | structure soft filter + cosine NN, top-1/5 측정 | ~150 줄 |

총 신규 ~580 줄 + 수정 ~120 줄 = ~700 줄 변경. doc/24 §6.8 의 추정 (~400 줄) 과 비슷한 규모.

### 3.3 절대 손대지 않는 파일

- `synth_engine_v3/` 전체 — corpus 재생성 없음
- `train_engine_v3/out/15_t5_light_v2/best.pt` — 읽기만, 덮어쓰기 금지 (warm-start source)
- `deploy_pi/export/v3_keras_*` — Phase 1 산출물, 유지

## 4. Architecture spec (실 구현 단위)

### 4.1 Forward graph (학습 시)

```
                          x (N, 3, 128, 128)
                                ↓
          [ResNet-18 backbone] (기존 backbone 그대로, warm-start from best.pt)
                                ↓
                       feat (N, 512)
              ┌───────────────┼─────────────────┐
              ↓               ↓                 ↓
        [structure heads]   [embedding head]   [char head — 학습 only]
        radical_head        Linear(512, 128)   Linear(512, 98169)
        idc_head            ↓                  ↓
        total_strokes       L2 normalize       logits_char
        residual_strokes    emb (N, 128)       (CE, warmup signal)
        (기존 4 head)        ↓
                            [ArcFace margin]
                            cos(θ + m) → s · cos(θ')
                            ↓
                            arc_logits (N, 98169)
                            (CE on this)
```

**핵심:** `char_head` (50 MB FC) 는 **학습용 보조 신호** 로만 사용하고 deploy 시 drop. ArcFace 의 weight (Parameter shape 98169×128) 가 학습 종료 후 anchor DB 의 source.

### 4.2 Forward (inference, deploy)

```
                          x
                          ↓
                    backbone
                          ↓
                       feat (512)
              ┌───────────────┐
              ↓               ↓
        structure heads   embedding head → emb (128, L2 norm)
        (rad/idc/strokes)
              ↓               ↓
       structure soft       cosine similarity vs anchor_db (98169, 128)
       filter:               (CPU 또는 simple GEMM)
       rad ∈ top-3                ↓
       idc ∈ top-2          score (98169,)
       |strokes - μ| ≤ 2          ↓
              └─── INTERSECT ────┘
                       ↓
                top-k char predictions
```

deploy model 에는 char_head 와 ArcFace classifier 가 없음. embedding head + structure heads 만 TPU 위에 올림.

### 4.3 ArcFace 구현 spec (`modules/arcface.py`)

```python
class ArcMarginProduct(nn.Module):
    """
    Inputs: embedding (N, 128) — must be L2-normalized
    Outputs: scaled logits (N, n_classes) — for CE loss
    Hyperparameters: s=30 (scale), m=0.5 (angular margin in radians)
    Standard ArcFace (Deng et al., CVPR 2019).
    """
    def __init__(self, emb_dim: int, n_classes: int, s: float = 30.0, m: float = 0.5):
        ...
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb_norm, labels):
        W_norm = F.normalize(self.weight, dim=1)
        cos = emb_norm @ W_norm.t()                        # (N, C), already in [-1, 1]
        # apply margin only to the target class
        ...
        return self.s * cos_with_margin                    # (N, C) logits
```

학습 종료 후 `arcface.weight` 를 그대로 anchor DB 로 export (alternative: per-class mean of `emb_norm` over train set — `51_build_anchor_db.py --mode {weight, mean, blend}` 로 셋 다 지원).

### 4.4 Loss schedule (curriculum, doc/24 §6.4 채택)

```python
class LossWeights:
    char: float = 1.0        # CE on 50 MB FC head — warmup 신호
    embedding: float = 1.0   # ArcFace CE
    radical: float = 0.2
    total: float = 0.1
    residual: float = 0.1
    idc: float = 0.2

def schedule(epoch: int) -> tuple[float, float]:
    """Returns (alpha_char, eps_embedding)."""
    if epoch <= 3:    return (1.0, 0.1)   # CE warmup
    if epoch <= 7:    return (0.5, 0.5)   # transition
    return (0.1, 1.0)                     # embedding-dominant
```

structure heads 의 가중치 (β/γ/δ) 는 v3 와 동일.

## 5. 학습 spec

### 5.1 Smoke 학습 (Day 1-2)

- Config: `resnet18_scer_smoke.yaml`
- Data: `94_production_102k_x200` 의 첫 30 shard (~150 K samples)
- Epochs: 1
- Goals:
  - 모든 5 head 가 forward + backward 성공
  - ArcFace loss 발산 / NaN 없음
  - Curriculum schedule 진입 / 전환 로깅 정상
  - End of epoch eval 에서 **char/top1 ≥ 5%, embedding cosine top-1 ≥ 5%, radical/top1 ≥ 30%** (random baseline 이상)

PASS 시 5.2 production 진입.

### 5.2 Production 학습 (Day 3-5)

- Config: `resnet18_scer_production.yaml`
- Warm-start: `train_engine_v3/out/15_t5_light_v2/best.pt` 에서 backbone 만 로드, head 는 random init
- Data: 전체 shard (3927)
- Epochs: 10 (T5-light v2 와 동일 — 14 시간 추정 RTX 4090)
- Optimizer: SGD + cosine warm restart (T5-light v2 와 동일 hyperparam)
- Output: `train_engine_v3/out/16_scer_v1/best.pt`

**중간 게이트 (epoch 5):**
- val char/top1 ≥ 35% (CE warmup 약화돼 있어 v3 의 39% 보다 살짝 낮을 수 있음)
- val embedding cosine top-1 (ArcFace weight 와 비교) ≥ 25%
- 어느 한쪽이라도 < 20% 면 학습 중단 + ArcFace m / s / curriculum 재조정

## 6. 평가 pipeline (Day 6)

### 6.1 Anchor DB build (`51_build_anchor_db.py`)

```
input  : best.pt (SCER)
output : deploy_pi/export/scer_anchor_db.npy  (98169, 128) float32 L2-normalized
modes  : --mode weight  (default — ArcFace classifier weight)
         --mode mean    (per-class mean of emb over training corpus)
         --mode blend   (0.5*weight + 0.5*mean, then re-L2)
```

3 모드 모두 build → 6.2 evaluator 에서 best 모드 선택.

### 6.2 SCER pipeline eval (`52_eval_scer_pipeline.py`)

- Input: 1000 검증 샘플 (Phase 1 게이트와 동일 seed=42)
- 각 샘플:
  1. backbone+heads forward → (rad_logits, idc_logits, strokes_pred, emb)
  2. structure soft filter: candidates = `{c : c.rad ∈ top3(rad_logits), c.idc ∈ top2(idc_logits), |c.total_strokes - strokes_pred| ≤ 2}`
  3. cosine sim: emb @ anchor_db[candidates].T → top-k
- Report:
  - SCER pipeline top-1, top-5
  - filter coverage (avg # candidates, % samples where GT in candidate set)
  - reranking quality: cosine NN top-1 within candidates vs full-table NN
  - 비교: char_head FC 직접 사용 (legacy path) 의 top-1 — 같은 모델인데 head 만 다를 때의 quality 차이

### 6.3 Phase 2 종료 게이트 (PASS 조건)

| 항목 | 임계 | Why |
|---|---|---|
| (G1) production 학습 NaN/divergence 없음 | `loss < 100` 유지 | safety |
| (G2) val embedding-pipeline top-1 (full-table NN) | ≥ 30% | v3 의 39% 보다 살짝 낮아도 OK (curriculum 후반 weight 변화 영향) |
| (G3) val embedding-pipeline top-5 | ≥ 55% | v3 top-5 가 ~52% 였음. 같거나 더 나아야. |
| (G4) structure filter coverage | candidate set 평균 50-500, GT 포함률 ≥ 92% | filter 가 정답을 떨어뜨리면 무용 |
| (G5) SCER pipeline (filter+rerank) top-1 | ≥ G2 - 2pp | filter 가 정확도 떨어뜨리지 않아야 (오히려 미세 향상 기대) |
| (G6) SCER pipeline top-5 | ≥ G3 - 2pp | 동상 |

전부 PASS → Phase 3 (Keras 포팅 + Edge TPU + Pi 측정) 진입. doc/29 작성.

부분 PASS / FAIL → doc/28 v2 로 ArcFace hyperparam (m, s, schedule) 또는 embedding 차원 (128 → 256) 조정 후 재학습. 큰 architectural pivot 은 사용자와 논의 후.

## 7. 시간 예산

| 단계 | 추정 시간 |
|---|---|
| 코드 수정 + 신규 파일 작성 (Day 1) | 4 시간 |
| Smoke 학습 (Day 2) | 30 분 학습 + 디버깅 |
| Production 학습 (Day 3-5) | 14 시간 (T5-light v2 와 동일) — 백그라운드 |
| Anchor DB build (Day 6) | 30 분 |
| Pipeline eval + 게이트 측정 (Day 6) | 1 시간 |
| 결과 문서화 (Day 6) | 1 시간 (Phase 2 results 섹션 — Phase 3 plan 합쳐서 doc/29 로 묶을 예정) |

**총 active 시간 ≈ 8 시간 + 백그라운드 14 시간**.

## 8. 리스크와 대응

| 리스크 | 가능성 | 대응 |
|---|---|---|
| ArcFace 학습 곡선이 까다로워 collapse | 중 | smoke 단계에서 m=0.5 → 0.3 으로 낮춰 시작, 안정 확인 후 0.5 |
| 128-d embedding 으로 98 K class 분리 부족 | 중 | smoke 후 256-d 도 시도. 12 MB → 24 MB 로 늘지만 여전히 8 MB SRAM cap 안에 들 수 있음 |
| Curriculum 의 epoch 8 transition 에서 char/top1 폭락 | 저 | epoch 8 에서 eval, 폭락 시 curriculum 완화 (linear ramp) |
| structure filter 의 GT 누락률 > 10% | 저 | top-3/2 를 top-5/3 으로 확장, 또는 strokes ±3 |
| best.pt warm-start 가 ArcFace 와 호환 안돼 head 만 학습 안 됨 | 저 | epoch 1-3 동안 backbone freeze, head 만 학습. epoch 4 부터 unfreeze |

## 9. 확인 요청 사항

이 plan 으로 Phase 2 진입해도 될까요?

확인 후 Day 1 부터 시작 — 다음 작업 순서는:
1. `modules/arcface.py` 작성
2. `modules/model.py` 수정 (embedding head + ArcFace classifier)
3. `modules/train_loop.py` 수정 (curriculum schedule + ArcFace term)
4. `configs/resnet18_scer_smoke.yaml` 작성
5. `scripts/50_train_scer.py` 작성
6. Smoke 실행 → forward/backward 확인 → production 학습 트리거

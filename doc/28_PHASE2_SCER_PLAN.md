# Phase 2 — SCER 학습 plan (v2: train_engine_v4 신규 생성)

작성: 2026-04-28
선행: doc/24 §6 (SCER 제안), doc/27 (Phase 1 deploy 경로 확정)
후행: doc/29 (예정 — Phase 2 결과 + Phase 3 Keras 포팅)

> **v1 → v2 pivot (2026-04-28):** v1 plan 은 `train_engine_v3/modules/` 의
> model.py / train_loop.py 를 직접 mutate 하는 안이었음. 사용자 지적: v3 는
> "확정" 상태 (best.pt + T5-light v2 baseline) 라 mutate 하면 frozen 자산이
> 흐려진다. 또한 프로젝트 컨벤션 (synth v1→v2→v3, train v1→v2→v3) 과
> doc/24 §6.8 의 원안이 모두 `train_engine_v4/` 를 전제. v2 는 이를 반영해
> **train_engine_v3 read-only, train_engine_v4 신규 생성** 으로 재계획.

## 0. 한 줄 요약

`train_engine_v4/` 디렉토리를 신규로 만들어 SCER (backbone + 3 small heads + 128-d embedding head + ArcFace) 모델을 학습한다. v3 의 backbone weight 만 warm-start 로 활용하고 v3 코드는 손대지 않는다. 공유 모듈 (shard_dataset / aux_labels / sysmon / utils) 은 `from train_engine_v3.modules import ...` 로 import 재사용.

## 1. v4 신규 생성 결정의 근거

| 기준 | 결과 |
|---|---|
| v3 의 "확정" 자산 (best.pt) 보존 | ✅ read-only, warm-start 만 |
| 프로젝트 컨벤션 (vN → vN+1) | ✅ 따른다 |
| Architecture 분리 (FC-based vs embedding-based) | ✅ 패러다임이 다른 모델은 디렉토리 분리 |
| doc/24 §6.8 가 명시한 경로 (`train_engine_v4/...`) | ✅ 일치 |
| 공유 모듈 중복 회피 | ✅ v3 에서 import (shard_dataset / aux_labels / sysmon / utils) |
| 실 작성 코드량 | +280 줄 정도 더 (model.py, train_loop.py 재작성) — frozen 보존 가치보다 작음 |

## 2. 비목적 (out of scope, doc/29 로 이월)

- Keras 포팅 (`keras_resnet18.py` 확장: SCER 의 embedding head + structure heads 추가)
- Edge TPU INT8 변환 + 컴파일
- Pi 실측 latency
- canonical_v3 DB 의 (rad, idc, strokes) → chars reverse index 사전 빌드

이 4 항목은 Phase 2 학습 결과가 게이트를 통과한 후 진행. 미리 해놓아도 학습 실패 시 무용지물.

## 3. 디렉토리 구조 + 파일 spec

### 3.1 `train_engine_v4/` 신규 트리

```
train_engine_v4/
├── __init__.py                     # 빈 파일
├── modules/
│   ├── __init__.py                 # 빈 파일
│   ├── model.py                    # 신규  ~150 줄 — SCER 모델
│   ├── train_loop.py               # 신규  ~250 줄 — curriculum + ArcFace term
│   └── arcface.py                  # 신규   ~70 줄 — ArcMarginProduct
├── configs/
│   ├── scer_smoke.yaml             # 신규   ~30 줄 — 4 shards × 1 epoch sanity
│   ├── scer_throughput.yaml        # 신규   ~30 줄 — 50 shards × 1 epoch img/s 측정
│   └── scer_production.yaml        # 신규   ~50 줄 — full corpus × 10 epoch
├── scripts/
│   ├── 00_smoke.py                 # 신규  ~150 줄 — 1 epoch sanity
│   ├── 50_train_scer.py            # 신규  ~200 줄 — production 학습
│   ├── 51_build_anchor_db.py       # 신규   ~80 줄 — (98169, 128) anchor table
│   └── 52_eval_scer_pipeline.py    # 신규  ~150 줄 — soft filter + cosine NN
└── out/                            # gitignored (.gitignore 의 train_engine_*/out/ 적용)
    └── 16_scer_v1/                 # production 학습 산출물
```

총 신규 ~1130 줄.

### 3.2 v3 측 import (재사용, 변경 0)

v4 의 학습 / 평가 코드는 v3 의 다음 모듈을 import 해서 그대로 사용:

```python
# train_engine_v4/scripts/00_smoke.py 등에서
from train_engine_v3.modules.shard_dataset import (
    TensorShardDataset, build_shard_train_val_split,
    build_stratified_val_split, list_shards,
)
from train_engine_v3.modules.aux_labels import AuxTable
from train_engine_v3.modules.sysmon import SysMon
from train_engine_v3.modules.utils import ...
```

shard 포맷, aux label 포맷, sysmon 모두 동일하므로 forking 이유 없음. v3 의 한 모듈에서 버그가 발견되면 v3/v4 양쪽이 동시에 fix 받음.

### 3.3 v4 가 v3 best.pt 를 warm-start 하는 방식

```python
# train_engine_v4/modules/model.py 의 build_scer() 내부
def load_v3_backbone(scer_model, v3_ckpt_path: Path) -> None:
    state = torch.load(v3_ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    backbone_only = {k[len("backbone."):]: v
                     for k, v in sd.items() if k.startswith("backbone.")}
    scer_model.backbone.load_state_dict(backbone_only)
```

ArcFace classifier / embedding head / aux head 들은 random init.

### 3.4 v3 read-only 보장

Phase 2 동안 다음 파일은 절대 수정 / 덮어쓰기 금지:
- `train_engine_v3/out/15_t5_light_v2/best.pt` (warm-start source)
- `train_engine_v3/modules/*.py` (import target)
- `train_engine_v3/configs/*.yaml`
- `train_engine_v3/scripts/*.py`
- `deploy_pi/export/v3_keras_*` (Phase 1 산출물)
- `synth_engine_v3/` 전체

## 4. SCER architecture (구현 단위)

### 4.1 학습 시 forward graph

```
                          x (N, 3, 128, 128)
                                ↓
          [ResNet-18 backbone] (warm-start from v3 best.pt)
                                ↓
                       feat (N, 512)
              ┌───────────────┼─────────────────┐
              ↓               ↓                 ↓
        [structure heads]   [embedding head]   [char head — 학습 only]
        radical_head        Linear(512, 128)   Linear(512, 98169)
        idc_head            ↓                  ↓
        total_strokes       L2 normalize       logits_char (CE warmup)
        residual_strokes    emb (N, 128)
        (4 head 그대로)      ↓
                            [ArcFace margin]
                            cos(θ + m) → s · cos(θ')
                            ↓
                            arc_logits (N, 98169)
                            (CE on this)
```

**핵심:** char_head (50 MB FC) 는 학습 보조 신호로만 사용, deploy 시 drop. ArcFace 의 weight (Parameter shape 98169×128) 가 학습 종료 후 anchor DB source.

### 4.2 Inference / deploy 시 forward

```
              x → backbone → feat
              ┌───────────────┐
              ↓               ↓
        structure heads   embedding head → emb (128, L2 norm)
              ↓               ↓
       structure soft    cosine sim vs anchor_db (98169, 128)
       filter:                ↓
       rad ∈ top3       score (98169,)
       idc ∈ top2             ↓
       |strk-μ| ≤ 2     top-k restricted to candidates
              └─── INTERSECT ─┘
                       ↓
                top-k char predictions
```

deploy 모델 = backbone + embedding head + 3 structure heads. char head + ArcFace classifier 둘 다 drop.

### 4.3 ArcFace 구현 (`modules/arcface.py`)

```python
class ArcMarginProduct(nn.Module):
    """Embedding (N, D, L2-normalized) + label → scaled logits (N, C).
       s=30, m=0.5, standard ArcFace (Deng et al., CVPR 2019)."""
    def __init__(self, emb_dim, n_classes, s=30.0, m=0.5):
        ...
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, emb_norm, labels):
        W_norm = F.normalize(self.weight, dim=1)
        cos = emb_norm @ W_norm.t()
        # angular margin only on target column
        ...
        return self.s * cos_with_margin
```

학습 종료 후 `arcface.weight.detach()` 가 그대로 anchor DB. 또는 per-class mean of `emb_norm` over corpus 와 blend (51_build_anchor_db.py 의 `--mode {weight, mean, blend}`).

### 4.4 Curriculum schedule (5-tuple, 통합 발산 관리)

throughput 실측 시 `easy_margin=false + m=0.5` 를 random-init 새 head 부터 적용했더니 발산 (loss 13.3 → 19.5, arc loss saturate ~48). 원인: 랜덤 weight 에서 cos(θ) ≈ 0 → cos(θ + 0.5) ≈ -0.48 → 정답 logit < 오답 → CE 폭주. 그리고 동시에 lr=0.05 로 backbone 까지 두들기면 유일한 안정 자산 (v3 best.pt) 이 부서질 위험. 따라서 **m / easy_margin / backbone_trainable 모두 epoch 별로 ramp** 하는 통합 schedule 로 변경.

```python
@dataclass
class LossWeights:
    char: float = 1.0      # CE on 50 MB FC head — warmup 신호
    embedding: float = 1.0 # ArcFace CE
    radical: float = 0.2
    total: float = 0.1
    residual: float = 0.1
    idc: float = 0.2

def schedule(epoch: int) -> tuple[float, float, float, bool, bool, str]:
    """Returns (alpha_char, eps_embedding, arc_m, easy_margin,
                backbone_trainable, phase_name)."""
    if epoch <= 3:    return (1.0, 0.1, 0.3, True,  False, "warmup")
    if epoch <= 7:    return (0.5, 0.5, 0.4, False, True,  "transition")
    return                  (0.1, 1.0, 0.5, False, True,  "fine")
```

| Epoch | α (CE) | ε (Arc) | m (margin) | easy_margin | backbone_trainable | phase |
|---:|---:|---:|---:|---:|---:|---|
| 1-3 | 1.0 | 0.1 | 0.3 | true  | false (freeze) | warmup |
| 4-7 | 0.5 | 0.5 | 0.4 | false | true           | transition |
| 8+  | 0.1 | 1.0 | 0.5 | false | true           | fine |

**왜 이렇게:**
- **m ramp (0.3 → 0.4 → 0.5)**: random init 에서 cos(θ + 0.5) 발산 회피. 작은 margin 으로 시작해서 head 가 의미있는 cos(θ) 값을 가질 때 큰 margin 적용.
- **easy_margin warmup (true → false)**: easy_margin 은 cos(θ) ≤ 0 영역에서 margin 안 적용 → 초기 stable, 후기에 standard ArcFace 로.
- **backbone freeze epoch 1-3**: 새 head (embedding / arc / char) 가 정렬 시점에 warm-started backbone (v3 best.pt) 을 큰 gradient 로 망치지 않도록. epoch 4 부터 unfreeze.
- **structure heads 는 항상 train**: warm-started 라 안정. (β, γ, δ 가중치는 v3 와 동일)

`print_epoch_start(epoch)` 가 epoch 시작 시 `model.arc_classifier.set_margin(m)`, `model.arc_classifier.easy_margin = e`, `model.set_backbone_trainable(backbone_trainable)` 호출. trainable param count 도 함께 로그.

### 4.5 Non-finite loss / gradient guard

학습 stability 의 마지막 안전망. **두 단계** — pre-backward (loss check) + post-backward (grad check). 17h 학습 도중 한 번이라도 NaN 이 흘러 들어오면 전체 학습이 corrupt 되므로 두 단계 모두 필수.

**Sliding-window rate-based (revised after first-launch abort, 2026-04-28)**:

첫 production launch 가 epoch 1 의 step 18,747 (60% 진행) 에서 cumulative `MAX_NAN_STEPS=10` 도달로 abort. 그러나:
- 그 시점 loss=3.589 (정상 범위), 학습 13.3 → 5.85 단조 감소
- nan rate = 10 / 18,747 = **0.05%** (sanity 의 2% 보다 *훨씬* 낮음)
- AMP GradScaler 의 자연스러운 underflow 처리의 부산물

→ Cumulative 절대값 threshold 가 **step-수에 sensitive** 해서 sanity (200 step) 와 production (300k step) 에 같은 threshold 적용 불가. **Sliding window 로 변경**.

```python
NAN_WINDOW_SIZE = 1000           # 최근 N step
NAN_RATE_ABORT = 0.05            # 5% (= 50 / 1000) → abort
NAN_RATE_WARN = 0.01             # 1% (= 10 / 1000) → log warn
```

매 step 에서 `nan_window` (deque maxlen=1000) 에 outcome 기록 (OK=0, skipped=1). `len == 1000` 이고 `sum >= 50` 이면 abort. 이 design 은:
- *systemic* instability 만 잡음 (sustained inability to make progress)
- AMP transient overflow (정상 동작) 는 통과
- step-수 무관하게 일관

`nan_window` 는 epoch 경계를 *건너뛰며 carry-over* (Codex review #2 #2): caller 가 `nan_window_in=...` 으로 직전 epoch 의 deque 를 다음 epoch 에 전달.

### 4.6 Step-level intra-epoch checkpoint (Codex review #3)

첫 launch 의 abort 는 epoch 1 끝나기 전이라 `last.pt` 가 *한 번도 저장 안 됨* → 50분 학습 손실. fix: `ckpt_every_steps=5000` 로 epoch 안에서도 last.pt 자동 저장.

```python
# train_loop.py 안에
if (i + 1) % ckpt_every_steps == 0:
    ckpt_callback(i + 1)         # caller-provided closure (atomic save)

# 또한 abort 직전에도 한 번 더 save
if abort_threshold_reached:
    ckpt_callback(i + 1)
    raise RuntimeError(...)
```

`50_train_scer.py` 가 closure 를 만들어서 model/optimizer/scaler/scheduler/metrics/nan_count 모두 캡처. resume 시 `step_in_epoch` 도 읽음 (다만 dataloader 의 정확한 step 위치는 못 복원하므로 epoch 처음부터 다시 — 단, 가중치는 mid-epoch 상태 유지).

**Pre-backward (loss):**

```python
if not torch.isfinite(loss):
    nan_count += 1
    optimizer.zero_grad(set_to_none=True)
    log_components(losses)              # which head produced NaN
    if nan_count >= MAX_NAN_STEPS:      # default = 10
        raise RuntimeError("aborted: too many non-finite losses")
    continue                             # backward + step 모두 skip
```

**Post-backward (grad — Codex review §4.5 #2 반영):**

`loss` 가 finite 인데 grad 만 NaN/Inf 나오는 경로 존재 (AMP underflow, ArcFace cos saturate 후 sqrt(1 - cos²) ≈ 0 분할, embedding L2-norm 의 0/0). AMP 사용 시:

```python
# AMP 일 때: scaler 로 backward → unscale → clip 순서 필수
scaler.scale(loss).backward()
scaler.unscale_(optimizer)               # grad 를 fp32 scale 로 되돌림 (이 후 clip 가능)

try:
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0,
        error_if_nonfinite=True,         # NaN/Inf 면 RuntimeError
    )
except RuntimeError:
    nan_count += 1
    optimizer.zero_grad(set_to_none=True)
    log_components(losses)
    scaler.update()                       # 이 step 의 inf check 가 scaler 의 backoff trigger
    if nan_count >= MAX_NAN_STEPS:
        raise RuntimeError("aborted: too many non-finite gradients")
    continue                              # optimizer.step skip

scaler.step(optimizer)                    # finite 보장된 후에만 step
scaler.update()
```

bf16 또는 fp32 일 때는 `scaler` 없이:
```python
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
optimizer.step()
```

**MAX_NAN_STEPS=10**: 일시적 단발성 NaN 은 step skip 으로 통과 (AMP scale 이 자동으로 줄어들면 다음 step 부터 회복), 누적 10 회 초과면 systemic 발산이라 abort.

**Sanity 단계 (B) 에서 강제 NaN test 1 회**: debug flag (`--inject-nan-step N`) 로 N 번째 step 에서 loss 에 NaN 주입, guard 가 정확히 step skip + abort 까지 가는지 확인. 17.5h job 트리거 전 마지막 검증.

## 5. 학습 spec

### 5.1 Smoke 학습 (Day 1-2)

- Config: `train_engine_v4/configs/scer_smoke.yaml`
- Script: `train_engine_v4/scripts/00_smoke.py`
- Data: `synth_engine_v3/out/94_production_102k_x200` 의 첫 30 shard (~150 K samples)
- Epochs: 1
- Goals (plumbing + warm-start + signal flow):
  - 모든 6 loss component (char, arc, rad, tot, res, idc) finite, NaN 없음
  - 학습 loss 단조 감소 (smoke 종료 시 epoch 1 시작 대비 -3 이상)
  - Curriculum schedule 진입 / 전환 로깅 정상 (epoch 1 = warmup α=1.0 ε=0.1)
  - 실시간 로그 spec (§9.1) 모든 필드 정상 출력 (timestamp / per-head / α/ε / throughput / sysmon)
  - v3 best.pt warm-start 정상: backbone + 4 structure head 로드, embedding/arc/char 는 random init
  - radical/top1 ≥ 25%, idc/top1 ≥ 40%, stroke_mae ≤ 4.0 (warm-start 살아있는지 확인. val=100 노이즈 큼)
  - char/top1, emb/top1 은 이 단계에서 의미 있는 절대값 측정 안 됨 (새 head 랜덤 초기화 + 67 step). random baseline (0.001%) 대비 1% 이상이면 signal 흐름 OK 로 간주.

  Smoke 의 절대 정확도 임계 (char/emb ≥ 5%) 같은 quality 검증은 production
  학습 (5.2) 에서 측정. smoke 는 architecture + 코드 + 로깅 검증이지 학습 quality 검증이 아니다.

### 5.2 Production 학습 (Day 3-5)

- Config: `train_engine_v4/configs/scer_production.yaml`
- Script: `train_engine_v4/scripts/50_train_scer.py`
- Warm-start: `train_engine_v3/out/15_t5_light_v2/best.pt` 의 backbone 만
- Data: 전체 shard (3927)
- Epochs: 10 (T5-light v2 와 동일)
- Wall-clock: **~17.5 시간** RTX 4080 Laptop GPU (`scer_throughput.yaml` 실측: batch=640 에서 3,120 img/s steady-state. v3 의 ~3,890 대비 80% — ArcFace 12M param GEMM + cos sim margin scatter 가 25% slowdown 의 주 원인. ±10% wall-clock 마진 가정).
- Optimizer: SGD + cosine warm restart (T5-light v2 hyperparam 그대로)
- Output: `train_engine_v4/out/16_scer_v1/best.pt`

**중간 게이트 (epoch 5):**
- val char/top1 ≥ 35%
- val embedding cosine top-1 (ArcFace weight 와 비교) ≥ 25%
- 어느 한쪽 < 20% → 학습 중단 + ArcFace m / s / curriculum 재조정

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
  - filter coverage (avg # candidates, % samples with GT in candidate set)
  - reranking quality: cosine NN top-1 within candidates vs full-table NN
  - 비교: char_head FC 직접 사용 (legacy path) 의 top-1 — 같은 모델인데 head 만 다를 때의 quality 차이

### 6.3 Phase 2 종료 게이트 (PASS 조건)

| 항목 | 임계 | Why |
|---|---|---|
| (G1) production 학습 NaN/divergence 없음 | `loss < 100` 유지 | safety |
| (G2) val embedding-pipeline top-1 (full-table NN) | ≥ 30% | v3 39% 보다 살짝 낮아도 OK (curriculum 후반 weight 변화) |
| (G3) val embedding-pipeline top-5 | ≥ 55% | v3 top-5 ~52%. 같거나 더 나아야 |
| (G4) structure filter coverage | candidate 평균 50-500, GT 포함률 ≥ 92% | filter 가 정답 떨어뜨리면 무용 |
| (G5) SCER pipeline (filter+rerank) top-1 | ≥ G2 - 2pp | filter 가 정확도 떨어뜨리지 않아야 |
| (G6) SCER pipeline top-5 | ≥ G3 - 2pp | 동상 |

전부 PASS → Phase 3 (Keras 포팅 + Edge TPU + Pi 측정) 진입, doc/29 작성.
부분 PASS / FAIL → ArcFace hyperparam (m, s, schedule) 또는 embedding 차원 (128 → 256) 조정 후 재학습.

## 7. 시간 예산

| 단계 | 추정 |
|---|---|
| 디렉토리 + 코드 작성 (Day 1) | 4-5 시간 (mutate 안 보다 1 시간 더) |
| Smoke 학습 (Day 2) | 30 분 + 디버깅 |
| Throughput 측정 (Day 2) | 3 분 (`scer_throughput.yaml`) |
| **Sanity 재검증 — m curriculum + freeze (Day 2)** | 3 분 (5 shards × 4 epoch) |
| Production 학습 (Day 3-5) | **17.5 시간** (백그라운드, 실측 기반) |
| Anchor DB build (Day 6) | 30 분 |
| Pipeline eval + 게이트 측정 (Day 6) | 1 시간 |
| 결과 문서 (Day 6) | 1 시간 (doc/29 의 Phase 2 결과 섹션) |

**total active ≈ 8-9 시간 + 백그라운드 ~17.5 시간**.

## 8. 리스크와 대응

| 리스크 | 가능성 | 대응 |
|---|---|---|
| ArcFace 학습 collapse (random init + m=0.5) | **확인됨 (throughput run)** | §4.4 m curriculum (0.3 → 0.4 → 0.5) + easy_margin warmup + epoch 1-3 backbone freeze 로 default 화. §4.5 non-finite guard 가 마지막 안전망. |
| 128-d 가 98 K class 분리 부족 | 중 | smoke 후 256-d 시도. 12 → 24 MB 인데 8 MB SRAM 압박 |
| Curriculum epoch 4, 8 transition 에서 loss 점프 | 저 | epoch 4 (freeze→unfreeze), epoch 8 (margin / weight 변화) eval. 점프 시 linear ramp 로 완화 |
| structure filter GT 누락률 > 10% | 저 | top-3/2 → top-5/3, strokes ±3 |
| 17.5h 학습 도중 NaN 한 번이라도 흘러 들어옴 | 중 | §4.5 guard: `MAX_NAN_STEPS=10` 까지 step skip, 초과 시 abort. `clip_grad_norm_(10.0)`. |
| Throughput 17.5h 가 ±10% 보다 더 늘어남 | 저 | 실측은 50 shards 만 — full 3927 shards 시 IO contention 가능. epoch 1-2 wall-clock 으로 실시간 재추정. |
| v3 import 시 module path 충돌 (`train_engine_v3` 가 sys.path 에 없을 때) | 중 | 각 v4 entry script 의 상단에 `sys.path.insert(0, REPO)` 추가 (v3 의 00_smoke.py 패턴 동일) |

## 9. 실시간 로깅 요구사항 (필수)

v4 학습은 **v3 와 동일한 실시간 로그 동작** 을 그대로 유지한다. 학습 도중 사용자가 콘솔을 보면 아래 형식이 `log_every` 스텝마다 즉시 (버퍼링 없이) 흘러나와야 함. v3 `train_engine_v3/modules/train_loop.py:209-221` 의 패턴 + ArcFace / curriculum 정보 추가.

### 9.1 라인 형식 (per `log_every` 스텝)

```
[HH:MM:SS] step {i+1}/{total}  loss={avg:.4f}  \
  char={l_char:.3f} arc={l_arc:.3f} rad={l_rad:.3f} \
  tot={l_tot:.3f} res={l_res:.3f} idc={l_idc:.3f}  \
  α={alpha:.2f} ε={eps:.2f}  lr={lr:.4g}  \
  (cum={cum_rate:.0f} win={win_rate:.0f} img/s, t={elapsed:.0f}s, eta={eta:.0f}s)  \
  {sysmon_snapshot}
```

v3 와 비교한 추가 항목:
- `arc=...` — ArcFace loss component (학습 우세 신호)
- `α=... ε=...` — 현재 epoch 의 curriculum weight 값 (transition 추적용)

다른 항목 (loss / per-head / lr / throughput / eta / sysmon) 은 v3 와 동일.

### 9.2 구현 요건

| 요건 | 구현 |
|---|---|
| 버퍼링 없음 | `print(msg, flush=True)` (v3 와 동일) |
| 파일 동시 캡처 | v3 의 `streaming_log.setup_logging()` 을 entry script 상단에서 호출. `out/16_scer_v1/run.log` 로 동시 기록 |
| Rolling-window throughput | v3 의 `deque(maxlen=window_steps)` 패턴 그대로 |
| ETA | `(elapsed / step) × steps_left` (v3 와 동일) |
| Sysmon snapshot | v3 의 `format_snapshot(sysmon.snapshot())` 를 import 해서 그대로 사용 |
| NVTX range | v3 의 `_nvtx("...")` 패턴 그대로 (forward / loss / backward / optim 분리) |

### 9.3 추가 로깅 (epoch 경계)

매 epoch 시작 시:
```
[HH:MM:SS] === Epoch N/10 ===  schedule: α={alpha:.2f} ε={eps:.2f}  \
            (curriculum phase: {warmup|transition|fine})
```

매 epoch 종료 + eval 후:
```
[HH:MM:SS] epoch N done in {dt:.0f}s  train_loss={X:.4f}  \
            val: char/top1={X:.3f} char/top5={X:.3f}  \
                 emb/top1={X:.3f} emb/top5={X:.3f}  \
                 rad/top1={X:.3f} idc/top1={X:.3f} stroke_mae={X:.2f}
```

`emb/top1` `emb/top5` 는 ArcFace classifier weight 와의 cosine sim top-K (anchor DB 의 oracle 평가 — 학습 중 모니터링용).

### 9.4 검증

Smoke 학습 (Day 2) 시 위 형식이 콘솔에 흘러나오는지 사용자가 직접 눈으로 확인. 로그가 안 흘러나오거나 buffered 되면 즉시 정지하고 디버깅. 이 요건은 학습 진행 가시성과 직결되므로 game-stopper 임.

## 10. 진입 조건

승인 받았으니 (2026-04-28) 다음 순서로 시작:

1. `train_engine_v4/` 디렉토리 + `__init__.py` 생성
2. `modules/arcface.py` 작성
3. `modules/model.py` 작성 (backbone + embedding head + structure heads + ArcFace classifier + warm-start helper)
4. `modules/train_loop.py` 작성 (curriculum + ArcFace term + 기존 multi-task loss 통합 + §9 실시간 로깅)
5. `configs/scer_smoke.yaml` 작성
6. `scripts/00_smoke.py` 작성 (v3 의 `00_smoke.py` 를 import 패턴 + SCER head 분기로 변형, `streaming_log.setup_logging` 호출)
7. Smoke 실행 → §9 로그 형식이 실시간 흘러나오는지 사용자 육안 확인 → 통과 시 production config + 50_train_scer.py 작성 → production 학습 트리거

## 11. 구현 진행 상태 (2026-04-28 시점)

### 11.1 ✅ 완료

| 항목 | 위치 | 비고 |
|---|---|---|
| `train_engine_v4/` + `__init__.py` | 트리 | §3.1 매핑 |
| `modules/arcface.py` | 신규 ~140 줄 | `ArcMarginProduct.set_margin()` curriculum hook 포함 |
| `modules/model.py` | 신규 ~230 줄 | `SCERModel`, `load_v3_backbone()`, `set_backbone_trainable()`, `build_deploy_state_dict()` |
| `modules/train_loop.py` | 신규 ~280 줄 | 6-component loss, eval (char/top1, emb/top1), §9 realtime log |
| `configs/scer_smoke.yaml` | 4 shards × 1 epoch | smoke 통과 (모든 head finite, char/top1=2.0%, emb/top1=2.0%) |
| `configs/scer_throughput.yaml` | 50 shards × 1 epoch | **3,120 img/s steady-state @ batch=640** (v3 80%) |
| `scripts/00_smoke.py` | 신규 ~280 줄 | smoke + throughput 양쪽 entry |

### 11.2 ⚠️ 부분 구현 — fix 필요

| 항목 | 현재 | 갭 (§4.4 / §4.5 반영 필요) |
|---|---|---|
| `train_loop.py::schedule(epoch)` | `(α, ε)` 2-tuple | `(α, ε, m, easy_margin, backbone_trainable, phase)` 6-tuple 으로 확장 |
| `train_loop.py::print_epoch_start` | α/ε 만 출력 | `arc_classifier.set_margin(m)` + `easy_margin` 토글 + `model.set_backbone_trainable(...)` + trainable param 수 로그 추가 |
| `train_loop.py::train_one_epoch` | guard 없음 | non-finite loss skip + `MAX_NAN_STEPS=10` abort + `clip_grad_norm_(10.0)` |
| `00_smoke.py` 의 eval logic | `or (ep == n_epochs)` 가 yaml `eval.every_epoch=99` 무시 | 강제 path 제거. throughput 측정이 진짜로 eval-free 되도록 |
| `arcface.py::forward` (선택) | `(N, C)` full matrix 4 개 allocate | gather/scatter 패턴으로 변경. ~30% 메모리 절감, throughput ~5% 가능. 안 해도 학습 가능. |

### 11.3 ⏳ 미작성

| 항목 | 위치 | 트리거 시점 |
|---|---|---|
| `configs/scer_production.yaml` | full corpus, 10 epoch | A 단계 fix 후 |
| `scripts/50_train_scer.py` | production entry | 상동 |
| `scripts/51_build_anchor_db.py` | (98169, 128) anchor 추출 | production 학습 진행 중 background |
| `scripts/52_eval_scer_pipeline.py` | structure filter + cosine NN + filter coverage | 상동 |
| `doc/29` | Phase 2 결과 + Phase 4 plan | 학습 종료 후 |

### 11.4 진행 단계 + 검증 결과

#### Step A — train_loop + 00_smoke fix ✅ 완료 (2026-04-28)

§11.2 의 모든 항목 구현:
- `schedule(epoch)` → 6-tuple `(α, ε, m, easy_margin, backbone_trainable, phase)`
- `print_epoch_start(epoch, total, model)` → curriculum 적용 + trainable param 수 로그
- `train_one_epoch` → pre-loss + post-grad guard (AMP unscale → `clip_grad_norm_(error_if_nonfinite=True)` → step skip → `MAX_NAN_STEPS=10` 시 abort)
- `--inject-nan-step N` debug flag
- `00_smoke.py` 의 `or (ep == n_epochs)` eval 강제 path 제거
- `00_smoke.py` 의 `eval_disabled` 분기 (eval_every > n_epochs 일 때 모든 shard 학습용, val 분할 안 함)
- `schedule(n_epochs)[5]` (phase string) — index 변경 따라 fix

#### Step B — Sanity 재검증 ✅ 1차 통과 (2026-04-28, `out/00c_scer_sanity/`)

##### B-1 (1차 sanity — cosine 없음)

**Config**: `configs/scer_sanity.yaml` (epoch 4) — 5 shards × 4 epoch, batch=640, fp16 AMP, scheduler 없음.

**Curriculum 전이 검증** (epoch banner 직접 확인):

| Epoch | banner 라인 (실제 출력) |
|---|---|
| 1 | `α=1.00 ε=0.10 m=0.30 easy_margin=True  backbone_trainable=False (phase: warmup)     trainable=63.11M / total=74.29M` |
| 2 | `α=1.00 ε=0.10 m=0.30 easy_margin=True  backbone_trainable=False (phase: warmup)     trainable=63.11M / total=74.29M` |
| 3 | `α=1.00 ε=0.10 m=0.30 easy_margin=True  backbone_trainable=False (phase: warmup)     trainable=63.11M / total=74.29M` |
| 4 | `α=0.50 ε=0.50 m=0.40 easy_margin=False backbone_trainable=True  (phase: transition) trainable=74.29M / total=74.29M` |

backbone freeze 차 = 74.29 - 63.11 = **11.18M ≈ ResNet-18 backbone param 수** — freeze 정확히 작동.

**학습 안정성**: throughput run (`easy_margin=false + m=0.5` 처음부터) 의 13.3 → 19.5 발산 재현 안 됨.

| Epoch | train_loss | dt | 비고 |
|---:|---:|---:|---|
| 1 | 6.67 | 14s | warmup 시작 |
| 2 | 6.38 | 5s | warmup 안정 |
| 3 | 4.07 | 5s | warmup 끝 |
| 4 | 6.58 | 11s | transition jump (m / easy / freeze 동시 변경 영향) |

epoch 3 → 4 점프 +2.51 — §8 의 "Curriculum epoch 4, 8 transition 에서 loss 점프" 리스크가 실제로 발생. Production 학습은 같은 5 shards 가 아니라 3927 shards 라 step 수가 100 배 (~390 step → ~39000 step / epoch). 4 epoch 내 회복 안 보였지만 transition phase 가 epoch 4-7 (4 epoch) 이라 이론상 회복 시간 충분.

**Non-finite GRAD guard 동작 (자연 발생)**:
- epoch 1 step 1: `arc=19.012` (initial random ArcFace head saturation), grad NaN, skip
- epoch 2 step 2: `arc=0.967` 인데 grad NaN (AMP underflow 의심), skip
- epoch 4 step 3: `arc=3.306` 인데 grad NaN (transition 직후), skip
- epoch 4 step 5: `arc=11.010` 인데 grad NaN, skip

총 4 회 자연 발생 → 모두 step skip → 학습 정상 진행. `MAX_NAN_STEPS=10` 도달 안 함. **인공 NaN 주입보다 더 강한 guard 검증** (자연 발생 케이스가 실제로 잡힘).

**Eval skip 검증**: `eval_sec=0.0`, `metrics={}` — Codex [medium] #3 fix 정상 동작.

**1차 판정**: 발산은 없으나 epoch 3→4 jump (+2.51) 이 관찰됨. Codex 의 [medium] #3 ("cosine warmup_epochs=1 is far from epoch 4") 우려와 결합해 **2차 sanity 진행**.

##### B-2 (2차 sanity — cosine + warmup=1, 5 epoch, Codex review #1/#2/#3 fix 반영)

**Config**: `configs/scer_sanity.yaml` (epoch 5 + cosine warmup=1, production scheduler 동일).
**Entry**: `scripts/50_train_scer.py` 사용 (smoke 가 아니라 production-grade entry — scheduler / boundary ckpt / cumulative nan 모두 검증).

| Epoch | train_loss | dt | nan_epoch | 비고 |
|---:|---:|---:|---:|---|
| 1 | 7.46 | 19s | 1 | warmup, cosine lr ramp |
| 2 | 6.36 | 7s | 1 | warmup, lr ~0.05 |
| 3 | 3.45 | 7s | 0 | warmup, **boundary anchor `epoch_003.pt` 저장** |
| 4 | **2.88** | 15s | 2 | **transition (m 0.3→0.4 + easy true→false + freeze→trainable + ε 0.1→0.5)** — jump 사라짐. 2.88 < 3.45 → 오히려 감소. |
| 5 | 1.80 | 11s | 0 | transition, 학습 정상 진행 |

**핵심 발견 — cosine 이 transition jump 를 흡수**:

| | 1차 (no cosine) | 2차 (cosine + warmup=1) |
|---|---:|---:|
| Epoch 3 train_loss | 4.07 | 3.45 |
| Epoch 4 train_loss | **6.58 (+2.51 jump)** | **2.88 (감소)** |
| Epoch 5 train_loss | — | **1.80** |

원리: cosine warmup_epochs=1 는 epoch 1 의 lr 만 ramp 하는 게 아니라, total_steps 기반 cosine decay 곡선이라 epoch 4 진입 시점에 이미 lr 이 0.05 → 0.026 까지 떨어져 있음. transition 의 동시 변경 (m, easy_margin, ε, backbone_trainable) 이 *큰 lr × 큰 magnitude* 로 들어오는 것을 *작은 lr × 큰 magnitude* 로 자동 완화. Codex [medium] #3 의 우려가 실제로는 *반대 방향* 으로 도움됨.

**Codex review #1, #2, #3 fix 검증**:

- **#1 boundary anchor**: epoch 3 종료 시 `epoch_003.pt` 저장 확인. `last.pt` overwrite 와 별도로 immutable. epoch 4 가 망가지더라도 epoch 3 으로 복원 가능.
- **#2 cumulative nan**: `nan_count_total=4` (= sum([1,1,0,2,0])). per-epoch reset 안 됨. `last.pt` 에 `nan_count_total` 저장 → resume 시 복원. abort threshold 10 미달.
- **#3 cosine sanity 재검증**: 위 표 — production scheduler 정확히 같은 설정 (cosine warmup=1) 으로 transition 통과 후 회복까지 확인.

**최종 판정**: ✅ PASS — production 학습 진입 가능. 증거: `out/00c_scer_sanity/{run.log, train_result.json, epoch_003.pt, last.pt}`.

#### Step C — production config + entry 작성 ✅ 완료

`configs/scer_production.yaml` + `scripts/50_train_scer.py` 작성 완료. boundary anchor `[3, 7]`, cumulative nan, cosine warmup=1 모두 적용.

#### Step D — Codex 의 별도 review ✅ 통과 (#1/#2/#3 fix 반영 + 2차 sanity 통과)

#### Step D-bis — Throughput ROI 재평가 + IO 우려 검증 ✅

**Codex 추가 review (학습시간 단축 ROI)**:
- [high] throughput 측정 freeze 상태 — 검증 결과 **틀린 추론** (그때 코드에 freeze 없었음, 모든 param trainable 상태 측정).
- [medium] eval budget — **valid**. `val_per_shard=25 × 3927 = 98k samples / eval` (1500 아님). 이전 25 → 3 으로 변경. eval ~1.5분/epoch.
- [medium] gather/scatter ROI single-digit — 동의. skip.

**Disk IO 우려 검증** (small shards 의 throughput 이 full-corpus 와 다른가):
- shard 245 MB × 3927 = 899 GB. RAM 32 GB → 페이지 캐시 절대 불가.
- 그러나 v3 production (full 3927) 데이터: epoch 10 throughput **4,124 img/s**, smoke throughput 측정 **3,890 img/s** — 거의 일치. **NVMe SSD + DataLoader prefetch 로 IO 가 GPU forward 보다 빠름** → IO bound 아님.
- 결론: small-shard throughput 이 production 의 reasonable proxy. 추가 측정 불필요.

**Wall-clock 보정**: v3 production 13.7h × SCER ratio (3120/3890 = 0.8 → 1/0.8 = 1.25) = **~17h ± 10%**. doc 의 ~17.5h 추정과 일치. eval 보정 후 **~17h train + ~15분 eval = ~17.3h**.

#### Step E — Production 학습 1차 시도 (2026-04-28) → ❌ abort

**시작**: 16:51, throughput 매우 양호 (4,705 img/s steady-state, GPU 96%).
**진행**: epoch 1, loss 13.3 → 5.85 단조 감소, step 18,700/30,661 (61%).
**Abort**: 18:05:59, step 18,747 — `aborted: 10 non-finite gradients cumulative across run (limit 10)`

원인: §4.5 의 cumulative `MAX_NAN_STEPS=10` 이 너무 엄격. production 의 0.05% rate (정상 AMP underflow 의 자연 부산물) 만으로 도달. 학습 자체는 healthy. last.pt 도 epoch 1 끝나기 전이라 저장 안 됨 → **50분 학습 손실**.

#### Step E-bis — Guard 재설계 (Codex review #3 round)

§4.5 Cumulative count → **sliding-window rate** (위 §4.5 본문 갱신 참고).

**Codex finding fix 적용**:
- ✅ #1 sliding window 에 OK step 도 기록 (rate 가 의미 있게 계산됨)
- ✅ #2 `nan_window` epoch 경계 carry-over (return + nan_window_in arg)
- ✅ #3 step-level intra-epoch ckpt 추가 (§4.6 신설, `ckpt_every_steps=5000`)

#### Step E-redux — Production 학습 2차 시도 — **다음**

#### Step F — 학습 진행 중 51/52 작성 ✅ 완료

`51_build_anchor_db.py` (~135 줄) — ArcFace classifier weight → L2-norm anchor table .npy.
`52_eval_scer_pipeline.py` (~280 줄) — structure prefilter + cosine NN, filter coverage / SCER top-k / 전 path 비교.

#### Step F-bis — Live monitor 작성 ✅ 완료

`60_live_monitor.py` (~330 줄) — run.log 를 incremental tail + matplotlib FuncAnimation 으로 stock-style 그래프 (loss / throughput / lr+α/ε) 갱신. 사용자가 학습 중 별도 터미널에서 실행하면 새 step 마다 자동 업데이트.

#### Step G — 학습 종료 후 평가 + doc/29 작성

#### Step F — 학습 진행 중 51/52 작성

#### Step G — 학습 종료 후 평가 + doc/29 작성

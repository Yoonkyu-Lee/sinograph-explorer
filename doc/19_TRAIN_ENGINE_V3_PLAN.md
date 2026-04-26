# Train Engine v3 — 다중 head 구조 + Aux Labels Sidecar 계획

작성: 2026-04-23. 대상: `train_engine_v3/` 신규 구축, synth ↔ canonical ↔
train 세 축의 label 흐름 설계.

이 문서는 **무엇을 어떻게 담아서 어떻게 학습시킬지** 만 다룬다. 구체 API /
코드는 구현 시점에 `train_engine_v3/V3_DESIGN.md` + 각 모듈 docstring 에.

관련 문서:
- [16_STRUCTURE_AWARE_V3_PLAN.md](16_STRUCTURE_AWARE_V3_PLAN.md) — 구조-aware 아이디어 원안
- [17_CANONICAL_V3_PLAN.md](17_CANONICAL_V3_PLAN.md) — canonical DB Phase 1 / 1.5 / 2
- [18_FINAL_PRESENTATION.md](18_FINAL_PRESENTATION.md) Slide 5 — Level A / A+ / B 로드맵
- [synth_engine_v3/V3_DESIGN.md](../synth_engine_v3/V3_DESIGN.md) — synth 쪽 Follow-up 1–5

---

## 1. 배경

midpoint (v2 baseline) 는 **단일 head 소프트맥스** — `image → ResNet18 → 10,932-way
softmax → codepoint`. 단일 codepoint label 만 학습 signal. Slide 5 에서 정리한
실측 failure 4 종 (鑑 동계 혼동 / rare 저신뢰 / 상호 regression / compositional
blindness) 는 이 단순 구조의 한계.

v3 의 novelty 는 **backbone 이 아니라 supervision topology 에 있음**: canonical_v3
가 제공하는 4 가지 구조 label (radical / total_strokes / residual_strokes /
ids_top_idc) 을 parallel head 로 붙여 공유 feature 가 compositional 구조를 인코딩
하도록 강제.

## 2. 전체 데이터 흐름

```
                                 ┌──────────────────────────┐
                                 │ sinograph_canonical_v3   │
                                 │ ids_merged.sqlite        │
                                 │  ├ characters_ids        │
                                 │  └ characters_structure  │
                                 └─────────────┬────────────┘
                                               │ (build time, once per corpus)
                    ┌──────────────────────────▼──────────────────────────┐
                    │ 50_export_aux_labels.py                             │
                    │  입력: class_index.json (corpus 생성 결과)          │
                    │        ids_merged.sqlite                            │
                    │  선별: 학습 signal 만 (radical / 획수 / IDC)        │
                    │  출력: aux_labels.npz   (Level A)                   │
                    └─────────────┬───────────────────────────────────────┘
                                  │
 ┌─────────────────────┐          │            ┌──────────────────────┐
 │ synth_engine_v3     │          │            │ train_engine_v3      │
 │  shard-NNNNN.npz    │          │            │  startup:            │
 │  class_index.json   ├──────────┼───────────►│    aux_table = load  │
 │  corpus_manifest    │          │            │      (aux_labels.npz)│
 └─────────────────────┘          │            │  batch:              │
                                  │            │    img, char_y = ... │
                                  └───────────►│    aux = aux_table[  │
                                               │            char_y]   │
                                               │    5-head forward    │
                                               │    multi-task loss   │
                                               └──────────────────────┘
```

핵심: **synth shard 포맷 불변** (`images u8, labels i64`). aux 는 **corpus 생성
직후 sidecar 로 한 번 생성**하고 train 때 index lookup. canonical DB 는 train
runtime 에 접근하지 않음 → corpus 자기완결.

## 3. Level A — 다중 head 구조 (primary plan)

[doc/18 Slide 5](18_FINAL_PRESENTATION.md) 와 동일. 재정리:

**입력**: RGB **192×192×3**, `(pixel/255 − 0.5)/0.5` → `[−1, 1]`.

**Backbone**: `torchvision.models.resnet18(weights=None)` 그대로. 마지막
`fc` 제거, 512-d feature 를 5 개 head 에 broadcast.

**5 parallel Linear heads (on 512-d pooled feature)**:

| Head | 출력 | 손실 | 가중치 | 용도 |
|---|---:|---|---:|---|
| `char` | N_class logits | CE | 1.0 | 최종 codepoint 분류 (primary) |
| `radical` | 214 logits | CE | 0.2 | 214 강희 부수 |
| `total_strokes` | 1 scalar | MSE | 0.1 | 총획수 회귀 |
| `residual_strokes` | 1 scalar | MSE | 0.1 | 부수 외 획수 회귀 |
| `ids_top_idc` | 12 logits | CE | 0.2 | 12-way 레이아웃 (⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻) |

**합산 손실**:
```
L_A = CE(char) + 0.2·CE(radical)
    + 0.1·MSE(total_strokes) + 0.1·MSE(residual_strokes)
    + 0.2·CE(ids_top_idc)
```

**Missing label 처리**: aux label 이 −1 (missing sentinel) 인 샘플은 해당 head
손실에서 mask out (loss *= valid_mask). char head 는 항상 유효.

**Inference**: aux head 전부 discard. `char` head 만 남겨 ONNX export →
기존 v2 배포 경로 (ONNX → INT8 TFLite) 그대로.

---

## 4. Aux Labels Sidecar 설계 (핵심)

### 4.1 포함 / 제외 기준

**포함**: 다음 3 조건 전부 만족하는 field 만.
1. **모델 loss 나 forward 에 직접 사용** (학습 signal).
2. **per-class deterministic** — 같은 char 의 모든 샘플에서 동일. sample-레벨
   정보는 이미 shard 의 char label 에 있으므로 중복 저장 금지.
3. **컴팩트 인코딩 가능** — int / float / bit. variable-length string / JSON 은
   학습 batch tensor 로 못 만드므로 제외.

**제외**:
- **Provenance / debug** — 어느 소스에서 왔나, 합의 수준, alternatives 등. DB 에
  남겨 두고 train runtime 에서 touch 하지 않음.
- **Variable-length string** — IDS 트리, agreement 레이블 문자열.
- **평가·조회 전용** — family / readings / meanings. 별도 파일 / 별도 inference-
  time display DB 로 분리.

### 4.2 Level A sidecar — 필드 정의

파일: `aux_labels.npz` — `class_index.json` 과 같은 디렉토리에 생성.

| key | dtype | shape | 내용 | Missing sentinel |
|---|---|---|---|---|
| `radical_idx` | int16 | `(N_class,)` | 0 – 214 (강희 부수) | −1 |
| `total_strokes` | int16 | `(N_class,)` | 0 – 84 (canonical_v3 실측 최대) | −1 |
| `residual_strokes` | int16 | `(N_class,)` | 0 – 84 | −1 |
| `ids_top_idc` | int8 | `(N_class,)` | 0 – 11 (`IDC_MAP` 순서) | −1 |
| `valid_mask` | uint8 | `(N_class, 4)` | 각 aux head 별 present flag (0/1) | — |
| `class_index_hash` | uint64 | `scalar` | 생성 시점의 class_index.json 해시 | — |

**IDC_MAP** (고정 상수):
```
⿰:0  ⿱:1  ⿲:2  ⿳:3  ⿴:4  ⿵:5
⿶:6  ⿷:7  ⿸:8  ⿹:9  ⿺:10 ⿻:11
```
(Unicode 16 확장 IDC 는 현재 스키마에서 −1 처리.)

**총 크기 (참고)**: N_class=10,932 → **78 KB**. N_class=76,000 → **548 KB**.
Train runtime 에 통째 GPU 메모리로 올림 — 무시할 만한 크기.

### 4.3 Level A+ 확장 — component multi-hot

Level A 정체 시 Level A+ 로 승격. 같은 `aux_labels.npz` 에 key 추가 (별 파일
만들지 않음):

| key | dtype | shape | 내용 |
|---|---|---|---|
| `components_multihot` | uint8 | `(N_class, K)` | K=512, 각 class 의 flat component 집합이 vocab 안에 있는지 (0/1) |
| `components_vocab` | int32 | `(K,)` | vocab 의 component codepoint 리스트 (label 해석용) |

**K=512 선정**: 빈도 상위 512 component. 세부 규정 [doc/17 §2.1 Phase 1.5]
(17_CANONICAL_V3_PLAN.md).

### 4.4 Flat component list 추출 — 소스 우선순위 **[중요]**

e-hanja online flat component list 는 **e-hanja class coverage 만** (~76 k).
나머지 코드포인트는 primary IDS 트리에서 추출해야 full 103 k coverage 달성.
따라서:

```
for each class c:
  if c has e-hanja flat component list:
      flat_components(c) = ehanja_components_json[c]
  else if c has primary_ids:
      flat_components(c) = strip_IDC(primary_ids[c])
      # IDC 12 문자 (⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻) 제거, 남은 codepoint list
  else:
      flat_components(c) = []                # 드묾, valid_mask=0
```

**`strip_IDC` 예시**:
- `⿰王恩` → `[王, 恩]`
- `⿱亠⿰金監` → `[亠, 金, 監]` (재귀 탈평면화)
- 원자 글자 (`永` 등) → `[]` (자기자신을 포함할지 여부는 구현 결정; Phase 1.5 에서 확정)

**선호 순서**:
1. **e-hanja flat 이 있으면 그걸 쓴다** — Korean 사전에서 손수 큐레이션된 decomposition. 한자 사전학 기준으로 신뢰도 높음.
2. **없으면 primary_ids 에서 flat 추출** — IDS 는 deterministic 재귀 decomposition. 76 k 이상 class 에 대해 이걸로 fallback.

둘의 decomposition 깊이가 다를 수 있음 (e.g. e-hanja 는 `['王','恩']`, IDS 는
재귀로 `[亠, 心, 口, 日, 玉]` 까지 내려갈 수도). 깊이 정책은 **"leaf 까지 다
flatten"** 으로 통일 (IDC 전부 제거, component 만 남김).

### 4.5 제외 확정

다음은 **sidecar 에 안 들어감**:

| field | 왜 제외 |
|---|---|
| `primary_ids` (IDS 트리 문자열) | variable-length. top-IDC 만 뽑아서 int 로 씀 |
| `primary_source` / `ids_sources_bitmask` | provenance |
| `agreement_level` / `has_struct_conflict` | provenance (추후 uncertainty weighting 에 쓰고 싶다면 `int8` 한 열 추가, Level A 에선 안 씀) |
| `ids_alternates_json` | debug 용 |
| `ehanja_components_json` 원본 string | vocab 매핑 거쳐 multi-hot 으로 저장 (Level A+) |
| `ehanja_agreement` | provenance |
| family_id / canonical_representative | **evaluation metric 용** (별도 `eval_meta.json`) |
| readings / meanings | **inference-time display** (canonical DB 그대로 조회) |

---

## 5. 데이터 흐름 — 세부

### 5.1 Synth 쪽 변화

**없음** (Phase 1 shard 포맷 그대로). 기존 `10_generate_corpus_v3.py` 가 생성하는:
```
out/NN_corpus/
  shard-00000.npz, shard-00001.npz, ...
  class_index.json
  corpus_manifest.jsonl
```
→ train 쪽에서 그대로 소비.

### 5.2 Canonical 쪽 변화

신규 스크립트 **`sinograph_canonical_v3/scripts/50_export_aux_labels.py`**:
- 입력: `--class-index PATH` (synth 가 만든 json) + `--db PATH` (ids_merged.sqlite)
- 출력: `--out aux_labels.npz` (기본 = class_index.json 과 같은 디렉토리)
- 동작: Level A 4 필드 + valid_mask + class_index_hash 저장
- 속도: 수 초 (row lookup × N_class)

Level A+ 승격 시 **`51_export_components.py`** 추가 (Phase 1.5 완료 후):
- vocab 선정 → `components_multihot` / `components_vocab` key 를 **기존
  aux_labels.npz 에 덧붙이기** (overwrite).

### 5.3 Train 쪽 변화 — train_engine_v3 신규 구축

**디렉토리** (전부 v2 에서 시작):
```
train_engine_v3/
  V3_DESIGN.md
  configs/
    resnet18_level_a_smoke.yaml       (1k class × 2 epoch — 동작 검증)
    resnet18_level_a_mini.yaml        (10,932 × 5 epoch — midpoint 재현)
    resnet18_level_a_full.yaml        (10,932 × 17 epoch — baseline)
    resnet18_level_a_ehanja.yaml      (76k × 20 epoch — production)
  modules/
    __init__.py
    model.py                         MultiHeadResNet18
    aux_labels.py                    npz → torch.Tensor (N_class, ...) 로 로드
    shard_dataset.py                 v2 와 동일 (이식)
    train_loop.py                    multi-task loss + per-head metrics
    sysmon.py                        v2 와 동일
    utils.py                         ckpt save/load 등
  scripts/
    00_smoke.py                      end-to-end smoke
    20_train.py                      main training driver
    21_eval.py                       held-out eval (per-head)
    22_export_onnx.py                char head only
    30_predict.py                    inference + canonical lookup
  out/                               (학습 산출물)
```

**`aux_labels.py`** 역할 (핵심):
- startup 에 `aux_labels.npz` 로드 → 6 개 key 를 torch.Tensor (long / float / bool)
  로 만들어 GPU 에 상주
- `class_index_hash` verify — shard 의 class_index.json 해시와 불일치면 에러
- 배치 forward 직전 `get_aux(char_y)` 호출 → `(radical, total, residual, idc, valid)`
  tuple 반환. 모두 `(B,)` 또는 `(B, 4)` tensor.

**`train_loop.py`** 변경 (v2 대비):
```python
logits = model(x)          # dict: char, radical, total, residual, idc
aux = aux_tbl[y]           # (radical, total, residual, idc, valid)

l_char   = CE(logits.char, y)
l_rad    = CE(logits.radical, aux.radical,   mask=aux.valid[:, 0])
l_total  = MSE(logits.total.squeeze(-1), aux.total,   mask=aux.valid[:, 1])
l_resid  = MSE(logits.resid.squeeze(-1), aux.resid,   mask=aux.valid[:, 2])
l_idc    = CE(logits.idc, aux.idc,       mask=aux.valid[:, 3])

loss = l_char + 0.2*l_rad + 0.1*l_total + 0.1*l_resid + 0.2*l_idc
```

**`train_engine_v3` 는 canonical_v3 를 직접 import 하지 않음**. sidecar npz 만
읽음 → dependency 감소.

---

## 6. 구현 순서

### 6.1 Phase T0 — train_engine_v3 부트스트랩
- [ ] `train_engine_v3/` 디렉토리 생성 + `V3_DESIGN.md` 초안 (개념·사용법·체크리스트)
- [ ] v2 모듈 중 변화 없는 것 복사: `shard_dataset.py` / `sysmon.py` / `utils.py`
- [ ] `model.py` — `MultiHeadResNet18` (backbone + 5 parallel Linear)
- [ ] `aux_labels.py` — npz 로드 + GPU tensor 상주 + `get_aux(y)` 인터페이스
- [ ] `train_loop.py` — multi-task loss + per-head metrics + valid_mask 처리

### 6.2 Phase T1 — Canonical export 스크립트
- [ ] `sinograph_canonical_v3/scripts/50_export_aux_labels.py` 작성
- [ ] 기존 corpus (`synth_engine_v3/out/80_production_v3r_shard256/`) 의
      class_index.json 으로 sidecar 생성, validity 검증
  - 4 head coverage ≥ 99 % 확인 (canonical_v3 Phase 1 수준)

### 6.3 Phase T2 — Smoke
- [ ] `configs/resnet18_level_a_smoke.yaml` (1k class × 500 샘플/class × 2 epoch)
- [ ] `scripts/00_smoke.py` — forward / loss / backward / ckpt save 검증
- [ ] 기대: loss 가 감소, aux head 가 meaningful accuracy 보임 (radical 부수
      top-1 > 50 %)

### 6.4 Phase T3 — v2 baseline 재현 (10,932 class)
- [ ] `configs/resnet18_level_a_mini.yaml` → 5 epoch 학습
- [ ] v2 baseline (92.82 % @ 17 epoch) 대비 **multi-task 가 같은 epoch 수에서
      더 높은 top-1 을 찍는지** 확인. 기대: +1 ~ +3 %p.
- [ ] radical / stroke / IDC head 의 aux accuracy 로 학습 signal 이 실제로
      backbone 에 들어가고 있는지 sanity check.

### 6.5 Phase T4 — Training resolution 192 전환
- [ ] `shard_dataset.py` gpu_transform 에 `F.interpolate(.. , size=192)` 삽입
- [ ] 기존 256 shard 를 그대로 decode 해서 192 로 resize
- [ ] 동일 config 에서 128 vs 192 top-1 비교. 기대: 고획수 (≥ 30 str.) tail 에서
      개선.

### 6.6 Phase T5 — Production (76 k class)
- [ ] class list 교체: T1 10,932 → e-hanja 76,013
- [ ] full_random_v3_realistic_v2 config 로 새 corpus 생성
- [ ] `configs/resnet18_level_a_ehanja.yaml`
- [ ] 학습 → 최종 demo 체크포인트
- [ ] ONNX export → INT8 TFLite → Pi 배포

### 6.7 Phase T6 — Level A+ (조건부)
- T3 + T5 에서 accuracy 정체 확인되면 진입. 아니면 skip.
- [ ] `sinograph_canonical_v3/scripts/51_export_components.py` — vocab + multi-hot
- [ ] `model.py` 에 `component_head: Linear(512 → 512)` 추가 + BCE loss
- [ ] `train_loop.py` 에 `L_component` + confusable-pair margin 추가
- [ ] 재학습 → 기존 failure pair (鑑/鍳, 媤/媳, 𤨒/璁) 에서의 confidence 변화
      측정

### 6.8 Phase T7 — Level B (발표 이후 / 논문용)
- radical-conditioned scoring fusion: `score(c|x) = z_char[c] + α · log p(r(c)|x)`
- 0 extra params. Level A ckpt 에서 inference-time 스위치만.

---

## 7. 체크포인트 / 파일 포맷

- `ckpt_epoch_{NN}.pt` — state_dict 전체 (backbone + 5 heads + optimizer +
  scheduler). 크기 ≈ 120 MB (10k class) / 270 MB (76k class).
- `ckpt_char_only.pt` — backbone + char head 만. 배포용 (ONNX export 전 단계).
  Level A aux head 는 discard. 크기 ≈ 67 MB (10k) / 320 MB (76k).

---

## 8. 평가 지표 (학습 중 / 최종)

**학습 중 (매 epoch)**:
- `char/top1`, `char/top5` — primary metric
- `radical/top1` — aux head sanity
- `total_strokes/mae`, `residual_strokes/mae` — regression aux
- `ids_top_idc/top1` — 12-way aux

**최종 (demo 용)**:
- 23-image real benchmark (midpoint 100 % / EasyOCR 43 % baseline 과 비교)
- Confusable-pair stress set (鑑 / 鍳 / 鐱 / 媤 / 媳 / 未 / 末 / 𤨒 / 璁 …)
- Family-aware accuracy — 예측 char 이 ground-truth char 의 canonical_representative
  family 에 속하면 부분점수. `eval_meta.json` 에 family 매핑 보관 (sidecar 밖).
- Pi CPU latency (< 500 ms target), Coral TPU latency (< 50 ms target).

---

## 9. Open Questions / Risks

1. **Aux head weight tuning** — 0.2 / 0.1 는 heuristic. 향후 Weighted-Sum /
   Uncertainty-Weighted (Kendall et al.) 로 튜닝 여지.
2. **76 k class 에서 char head 크기 폭발** — `Linear(512 → 76,013) = 39 M params`.
   class-embedding 분리 / tied classifier 같은 기법 고려 (Phase T5 에서 결정).
3. **Aux label coverage 구멍** — canonical_v3 의 99.9 %+ 이지만 100 % 아님.
   valid_mask 로 안전하게 처리. 혹시 coverage 구멍이 특정 block (예: Ext J)
   에 몰려 있으면 학습 신호 imbalance 발생 가능 — 모니터링 필요.
4. **Residual strokes 음수** — 일부 canonical row 에서 radical stroke 가
   잘못되어 total − radical = 음수 나오는 edge case. sanity clamp: `residual =
   max(0, total − radical)` 을 export 단계에서 확정.
5. **IDC = −1 비율** — atomic 문자 (自, 永, 金 등) 는 top-IDC 없음. valid_mask
   로 CE 에서 자동 제외. class 의 ~15 % 예상.

---

## 10. 한 줄 요약

> **학습 signal 만 sidecar npz 하나에 담는다. synth 는 shard 포맷 그대로,
> canonical 은 내부 원본 그대로, train 은 sidecar 만 보고 돌린다. e-hanja
> flat 이 있는 class 는 그걸 쓰고, 없으면 primary_ids 에서 IDC 벗겨 flat 으로
> derive 한다.**

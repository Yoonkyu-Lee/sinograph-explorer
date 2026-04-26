# Train Engine v3 — 설계 / 현황

상세 계획: [doc/19_TRAIN_ENGINE_V3_PLAN.md](../doc/19_TRAIN_ENGINE_V3_PLAN.md).
이 문서는 **v3 에서 지금 뭐가 돌아가고, 어떻게 쓰는지** 만 담는다.

## 한 줄

단일 head 소프트맥스 (v2) → **char 1 primary + 4 aux heads (radical / total /
residual strokes / ids_top_idc)** multi-task ResNet-18. Aux label 은 canonical_v3
에서 export 된 `aux_labels.npz` sidecar 에서 startup-load.

## 왜 v2 가 아니라 v3

v2 는 `image → codepoint` 단일 CE 손실. v3 는 동일 backbone 위에 aux head 4개
추가 → 공유 feature 가 구조 정보 (부수 / 획수 / IDC) 를 강제로 인코딩하도록.
Inference 때 aux head 전부 discard → 배포 latency 변화 없음.

자세한 head 표 / 손실식 / 왜 이 설계인지: [doc/19 §3](../doc/19_TRAIN_ENGINE_V3_PLAN.md#3-level-a--다중-head-구조-primary-plan).

## 디렉토리 구조

```
train_engine_v3/
  V3_DESIGN.md          이 문서
  configs/              YAML (smoke / mini / full / ehanja)
  modules/
    __init__.py
    model.py            MultiHeadResNet18 (backbone + 5 heads)
    aux_labels.py       aux_labels.npz 로드 + get_aux(char_y) 인터페이스
    shard_dataset.py    v2 copy — shard NPZ iterator (이식 그대로)
    sysmon.py           v2 copy — GPU/RAM monitor
    train_loop.py       multi-task train_one_epoch + evaluate
    utils.py            v2 copy — ckpt save/load 등
  scripts/
    00_smoke.py         end-to-end 1 epoch mini
    20_train.py         main training driver
    21_eval.py          held-out eval (per-head metrics)
    22_export_onnx.py   char head only → ONNX (aux discard)
    30_predict.py       inference + canonical display lookup
  out/                  학습 산출물 (NN_purpose/ 네이밍)
```

## 데이터 입력 경로

```
synth_engine_v3/out/NN_corpus/
  ├─ shard-00000.npz        ← (image u8, char_label i64)
  ├─ shard-00001.npz
  ├─ class_index.json        ← {notation: class_idx}
  └─ aux_labels.npz          ← sinograph_canonical_v3/50_export_aux_labels.py 로 생성
                               (radical_idx / total_strokes / residual_strokes /
                                ids_top_idc / valid_mask / class_index_hash)
```

train 은 이 3 파일만 touch. canonical DB / ehanja manifest 등엔 접근하지 않음 →
corpus 자기완결.

## 사용법 (개념)

```
# 1) corpus 생성 (synth_engine_v3 쪽)
python synth_engine_v3/scripts/10_generate_corpus_v3.py \
  --config synth_engine_v3/configs/full_random_v3_realistic_v2.yaml \
  --output-format tensor_shard \
  --out synth_engine_v3/out/NN_corpus

# 2) aux sidecar 생성 (canonical_v3 쪽)
python sinograph_canonical_v3/scripts/50_export_aux_labels.py \
  --class-index synth_engine_v3/out/NN_corpus/class_index.json \
  --out synth_engine_v3/out/NN_corpus/aux_labels.npz

# 3) 학습
python train_engine_v3/scripts/20_train.py \
  --config train_engine_v3/configs/resnet18_level_a_mini.yaml
```

## 현황 (2026-04-23)

| Phase | 상태 | 노트 |
|---|---|---|
| T0 bootstrap | ✅ | 디렉토리 + v2 모듈 이식 + model / aux_labels / train_loop 작성 |
| T1 canonical export | ✅ | `50_export_aux_labels.py`, 10,932 class coverage: radical 99.88 %, total 99.88 %, residual 99.81 %, idc 96.85 %. radical 은 1-indexed → 0-indexed 로 변환 |
| T2 smoke | ✅ | 4 shard × 1 epoch × 192 res. 5-head forward/backward/eval OK. char CE 9.4 → 수렴 방향, idc/top1 62.8 % after 23 steps. smooth_l1 로 regression loss 스케일 O(1) 유지 (MSE 였으면 7k 급 폭발) |
| T3 v2 baseline 재현 | pending | 10,932 × 5 epoch, multi-task vs single-task 비교 |
| T4 해상도 128 → 192 | 부분 | smoke 는 이미 192. full-run 시 v2 128 baseline 과 비교 |
| T5 production 76 k | pending | e-hanja class set |
| T6 Level A+ | 조건부 | accuracy 정체 시 |
| T7 Level B | 발표 이후 | radical-conditioned scoring fusion |

## 구현 메모

### Loss 스케일 정렬
doc/19 §3 는 stroke 두 head 에 MSE 를 지정했으나 실측 결과 target 이 1–84
범위라 MSE 가 target² ≈ 7k 급 폭발 → CE 항 (~5) 을 완전히 지배. **Smooth-L1
(Huber β=1.0)** 으로 교체 — semantically 동일 회귀, 스케일 O(target) 유지.

### Radical 인덱스
canonical_v3 DB 는 강희 부수 **1–214** (1-indexed). 학습 head 는 `Linear(512
→ 214)` 라 **0–213** 기대. 따라서 `50_export_aux_labels.py` 가 export 시점에
`−1` 해서 저장. train 쪽은 그대로 `nn.CrossEntropyLoss` target 으로 사용.

## v2 호환성 체크

- shard 포맷 동일 (`images u8[N,H,W,3], labels i64[N]`) — v2 shard 그대로 소비
- class_index.json 동일 format
- ckpt 포맷 다름 (v3 는 5 head state_dict)
- inference export 는 char head 만 남기므로 v2 배포 파이프라인 (ONNX → INT8
  TFLite → Pi) 재사용 가능

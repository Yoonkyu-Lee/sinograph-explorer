# 특정 한자 집중 강화 워크플로 (Targeted Fine-Tune)

배경: v3r_prod_t1 (10,932 class × 500 sample, val top-1 92.82%) 를 base 로 삼아,
**특정 한자 한 개** 를 더 강하게 학습시키는 mini-procedure. 풀 재학습 없이
경량 파인튜닝으로 target prob 를 끌어올린다. 2026-04-20 𤨒 (U+24A12) 케이스
로 검증.

## 왜 필요한가
Stage 2 production 모델은 모든 T1 class 를 500 sample 로 균등 학습. 개별 한자
중 특별히 robust 하게 인식시키고 싶은 경우 (demo character, 논문용 강조, 특정
고객 requirement 등) 전체 재학습 (~6시간) 은 비효율적. 가장 싼 방법부터 비싼
방법까지 정리:

| 방법 | 데이터 | 시간 | 부작용 | 추천도 |
|---|---|---:|---|---:|
| **Oversample fine-tune** (본 문서) | 새 1k sample + prod shard 일부 | **~30s** | 시각적 유사 class 에 regression 가능 | ★★★ |
| Class-weighted loss | 기존만 | 20 min + 로 재학습 | catastrophic forgetting 큼 | ★ |
| 풀 재학습 with 재분배 | 전체 + 추가 | ~6h | 없음 | ★★ (시간 여유 시) |

## 핵심 아이디어
1. **Oversample shard** — 대상 한자만 들어있는 mini-corpus (500-2000 sample)
   를 synth_engine_v3 로 재생성. label 은 prod class_index 의 target_idx 로
   remap.
2. **Anchor mini-batch** — 기존 prod shard 에서 4-10 개 무작위 추출. 이들이
   "non-target class 들도 잊지 말라" 는 regularization 역할.
3. **Mixed batch iteration** — 매 step 배치의 절반은 oversample, 절반은 anchor.
   단순 round-robin IterableDataset 로 구현.
4. **Low-LR brief tune** — LR 0.005, SGD+momentum, 200 step. 20 step 만에
   target accuracy 100% 도달 → 나머지는 prob 안정화.

## 절차 (𤨒 케이스 기준, 26초 wall-time)

### 1. class_list 작성 (1 line)
```jsonl
{"codepoint": "U+24A12", "char": "𤨒", "tiers": ["T1"], "tier_picked": "T1",
 "target_samples": 1000, "enriched_family_size": 1,
 "enriched_representative": "U+24A12", "block": "Ext_B_SMP",
 "source_flags": {"unihan": true, "ehanja_online": false}}
```

### 2. Oversample shard 생성 (~12s)
```bash
python synth_engine_v3/scripts/10_generate_corpus_v3.py \
  --config synth_engine_v3/configs/full_random_v3_realistic.yaml \
  --class-list class_list_24a12.jsonl \
  --pool union --strategy uniform \
  --mask-workers 4 --save-workers 2 --batch-size 256 \
  --output-format tensor_shard --shard-size 500 --shard-input-size 256 \
  --out d:/tmp/24a12_exp/shards --seed 4212
```
→ `shard-00000.npz`, `shard-00001.npz` (각 500 sample)

### 3. Label 리매핑 (prod class_index 기준)
shard 내 labels 는 local class_index 기준 (target_idx=0). prod class_index 기준
target_idx (𤨒 → 10719) 로 전부 덮어쓴다:
```python
labs = np.full_like(d["labels"], prod_ci["U+24A12"])  # 10719
np.savez(dst, images=d["images"], labels=labs)
```

### 4. Fine-tune (~26s)
`finetune_24a12.py` (d:/tmp/24a12_exp/) 핵심:
```python
# MixedIterable: ds_over 1 sample → ds_anchor 1 sample → 반복
ds_over   = TensorShardDataset(over_shards, ...)      # 2 shard × 500
ds_anchor = TensorShardDataset(rng.sample(all_prod, 4), ...)
mixed     = MixedIterable(ds_over, ds_anchor)
loader    = DataLoader(mixed, batch_size=256, num_workers=0)

opt   = SGD(lr=0.005, momentum=0.9, weight_decay=5e-4)
crit  = CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda")
# 200 step → target acc 100%, anchor acc 96-100% 유지
```

### 5. Before / After 검증 (held-out test renders)
prod best.pth vs best_ft.pth 로 동일 이미지 예측, target prob + rank 비교.

## 실험 결과 (𤨒, 200 step)

### Target boost
| render | before prob | after prob | Δ |
|---|---:|---:|---:|
| mingliub_idx0 | 40.0% | **91.0%** | +51.0 |
| mingliub_idx2 | 12.7% | **86.3%** | +73.6 |
| simsunb_idx0  | 36.0% | **86.1%** | +50.1 |

base 이미 rank 1, FT 후에는 압도적 confidence.

### 타 class 변화 (held-out, 전부 rank 1 유지 or 더 좋아짐 제외 1건)
| char | before | after | note |
|---|---:|---:|---|
| 畓 | 67.0% | 78.8% | ↑ spillover |
| 𤴡 | 5.8% | 26.0% | ↑ spillover (Ext B feature 전반 개선) |
| 中 | 33.1% | 43.8% | ↑ |
| 中(desat) | 41.3% | 49.7% | ↑ |
| 鑑 | 35.6% | 8.0% | rank 1 유지하나 하락 |
| **媤** | **4.6% rank 1** | **0.1% rank 6** | **regression** ✗ |

### EasyOCR 대조 실험 (base model 기준, 32_easyocr_compare.py)
| 이미지 | 우리 (base) | EasyOCR (5-lang) |
|---|---|---|
| 𤨒.webp (label 포함 실제 사진체) | 𤨒 33.4% ✓ | 璁, 琨, 瑠, 24A12 ✗ |
| 𤨒_mingliub | 𤨒 40.0% ✓ | 璁, 琨, 瑠 ✗ |
| 𤨒_mingliub_bold | 𤨒 12.7% ✓ | 璁, 琨, 瑠 ✗ |
| 𤨒_simsunb | 𤨒 36.0% ✓ | 璁, 琨, 瑠 ✗ |

우리 base 4/4 (100%), EasyOCR 0/4 (0%). EasyOCR 은 CJK Ext B 인식 자체가 없고,
시각적으로 가까운 玉-radical BMP (璁/琨/瑠) 로 반복 환각. **fine-tune 하지 않은
base 만으로도 이미 단독 승** — FT 는 정확도가 아니라 **confidence gap** 을
넓힌다 (top-1 40% → 91% 같은 식).

## 위험 요소 & 완화법
- **Visual-neighbor regression**: 𤨒 부스트 시 媤 가 rank 1 → 6 (시각적 layout
  공유). 다른 target 도 유사 peer set 가능성. **배포 전 holdout regression
  audit 필수**.
- 완화:
  - LR 더 낮게 (0.001) + step 늘려 평탄화
  - anchor shard 더 많이 (10+) 다양성 강화
  - EWC (Fisher-matrix weighted param regularization) 로 base 파라미터 deviation
    제한 → 구현 복잡도 ↑
  - 혹은 target + peer(媤) 두 class 를 같이 oversample

### Mutual-regression 교차 검증 (2026-04-20, 媤 실험)
같은 워크플로로 媤 (U+5AA4) 를 역방향 FT 한 결과, **𤨒 ↔ 媤 간섭이 대칭적** 임이
확인됨:

| 방향 | before | after |
|---|---|---|
| 𤨒 FT → 媤 | 4.62% rank 1 | 0.12% rank 6 |
| 媤 FT → 𤨒 | 40.0% rank 1 | 1.54% rank 2 |

두 한자는 feature space 에서 실제로 공간을 공유 (좌: 좁은 세로 편, 우: 복합
하단 구성). 한쪽을 밀면 반대쪽이 그만큼 밀림 — zero-sum.

추가 관찰: **媤 FT 는 side-effect 범위가 더 넓다**. 𤨒 FT 는 𤴡/中/鑑/畓 대부분
향상 (+10-20%p) 이었는데, 媤 FT 는 𤴡/中/鑑 전부 하락. 媤 의 radicals (女+思)
가 T1 에 빈번해 많은 class 경계에 관여하기 때문으로 추정. → **FT target 선정 시
그 한자의 radical 가 얼마나 common 한지 고려** 해야 함.

실사진 시집 시.png 은 4.62% → **61.20%** 로 +56.6%p 상승. 실제 deployment 관점
에서 가치 있는 개선. 자세한 before/after 는 `d:/tmp/5aa4_exp/report.md` 참고.

### 실용 guidance
- Visually-isolated target (𤨒 같은 rare Ext B) → regression 거의 없음, 안전
- Common-radical target (媤 의 女+思) → 광범위 regression, 신중히 또는
  multi-target co-training 고려
- 두 개 이상 target 이 동시에 필요하면 **multi-target oversample**
  (target_a + target_b + anchor 3-way mix) 로 co-train, 혹은 runtime 에
  쿼리 특성에 따라 FT 버전 switch 하는 load-on-demand 전략

## 재현 산출물
- `d:/tmp/24a12_exp/class_list_24a12.jsonl` — 1 class spec
- `d:/tmp/24a12_exp/shards/` — 원본 oversample (local idx)
- `d:/tmp/24a12_exp/shards_remapped/` — prod idx 로 relabel
- `d:/tmp/24a12_exp/finetune_24a12.py` — 본 파인튜닝 스크립트
- `d:/tmp/24a12_exp/best_ft.pth` — FT 후 체크포인트
- `d:/tmp/24a12_exp/report.md` — before/after 표
- `train_engine_v2/test_img/𤨒_*.png` — held-out test renders
- `train_engine_v2/out/03_v3r_prod_t1/easyocr_compare_24a12.jsonl` — EasyOCR 대조

## 다음 단계 (recommendation)
1. 이 워크플로를 `train_engine_v2/scripts/40_finetune_target.py` 로 정식 편입
   (현재는 d:/tmp 에 scratch 상태). CLI:
   `--target-cp U+XXXX --samples 1000 --steps 200 --lr 0.005`
2. EWC 또는 peer-aware oversample 로 regression 방지 옵션 추가
3. Multi-target (한 번에 여러 대상 부스트) 확장 — proposal demo 준비 용

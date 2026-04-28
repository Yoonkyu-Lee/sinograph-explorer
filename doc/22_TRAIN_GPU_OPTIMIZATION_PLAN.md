# Train Engine v3 — GPU 최적화 계획

작성: 2026-04-26.  대상: `train_engine_v3/`. corpus = 94_production_102k_x200
(98,169 class, 19.6M samples, 128 px shard).

> **진행 현황** (2026-04-26 갱신):
> - TG-0 베이스라인: ✅ Windows 3,373 img/s / WSL 2,668 img/s
> - TG-1: ✅ 측정 완료 — [doc/23](23_PHASE_TG1_RESULTS.md). winner = `bs=640 + cudnn`, **+20.9%** (4,078 img/s)
> - TG-2 / TG-3: pending

doc/20 의 synth-side GPU 최적화 (CPU PIL → CUDA polygon raster, 2.6× speedup)
와 같은 결의 train-side 계획. 단, **train pipeline 은 cuDNN/cuBLAS 가 대부분
dominant** 라 win 영역이 다름. 이 문서는 어디가 진짜 hot spot 이고, 어디에
custom CUDA kernel 이 의미 있고, 어디는 그냥 PyTorch native 로 충분한지 미리
정직하게 분류.

관련:
- [doc/19_TRAIN_ENGINE_V3_PLAN.md](19_TRAIN_ENGINE_V3_PLAN.md) — multi-head 설계
- [doc/20_GPU_OPTIMIZATION_PLAN.md](20_GPU_OPTIMIZATION_PLAN.md) — synth-side 최적화 사례
- [batched-lenet-cuda](https://github.com/Yoonkyu-Lee/batched-lenet-cuda/tree/main/archived) — ECE 408 수업 자료. 단 **train 에 적용가능한 lab 은 synth 와 다름**:
  - **L5 privatized atomics** — 98k-class histogram / 라벨 카운트 (global atomic 충돌 회피)
  - **L6 reduction** — fused multi-head loss 안의 block reduction
  - **L7 scan** — top-k / sampled-softmax bookkeeping (Level A+ 진입 시)
  - **L8 JDS / coalesced sparse** — 98k-class softmax 의 sparse gradient (Level A+ 검토)
  - **Project: kernel fusion + streams** — multi-head loss 융합, H2D 오버랩
  - **Lecture: Nsight Systems / NVTX** — 가장 ROI 높은 도구. 모든 stage 를 NVTX range 로 감싸서 Nsight 가 진짜 bottleneck 짚도록
  - L2/L3/L4 conv tiling, Tensor Core 응용은 **cuDNN 이 이미 커버** — train 에 적용 불가

---

## 1. Pipeline 분석 — step 단위 예상 비중

가설치 (RTX 4080 Laptop, AMP fp16, batch=128, 192 px):

| Stage | 추정 % | 주체 | 최적화 여지 |
|---|---:|---|---|
| (1) shard NPZ read + per-sample yield | 7-10% | DataLoader workers (CPU) | **HIGH** — 통째 batch GPU 로드로 대체 |
| (2) pin_memory + H2D copy | 3-5% | torch | **MED** — async + double-buffer |
| (3) uint8 → float + resize + normalize (gpu_transform) | 2-3% | torch | LOW — 이미 GPU |
| (4) ResNet18 backbone forward | ~35% | cuDNN | **None** (cuDNN dominant) |
| (5) Char head Linear(512→98169) fwd | ~5% | cuBLAS | None |
| (6) Aux 4 heads fwd | ~1% | cuBLAS | LOW — 워낙 작음 |
| (7) AuxTable.get_aux: 5 gather | ~0.5% | torch index | **MED** — 1 fused gather |
| (8) Multi-task loss (CE×3 + smooth_l1×2) | 2-3% | torch | **MED** — 1 fused kernel |
| (9) ResNet18 backward | ~30% | cuDNN | None |
| (10) Char head Linear bwd | ~5% | cuBLAS | None |
| (11) Optimizer step | ~1% | torch | None |

Custom CUDA 의 이론적 최대 win:
- (1) + (2) + (7) + (8) = 12-19% of step time
- 이 부분을 50-70% 줄이면 **end-to-end 6-13%** 빨라짐

PyTorch native 기법의 win (custom 안 써도):
- channels_last format: 1.2-1.5× (Tensor Core 가 NHWC 선호)
- torch.compile: 1.3-2× (graph fusion, inductor)
- bf16 vs fp16 AMP: 같은 속도, 수치 안정성↑
- cudnn.benchmark: 미세
- batch size up: VRAM 여유 따라 5-15%

→ **Free win 1.5-2×, custom CUDA 추가 1.06-1.13×, 합쳐 1.6-2.3× 기대**.

---

## 2. 단계별 계획

### Phase TG-0 — Smoke 정합성 + 베이스라인 + NVTX 계측

> **목표**: 옛 corpus 80_production_v3r_shard256 (256 px) → 새 corpus
> 94_production_102k_x200 (128 px) 로 smoke 가 도는지 검증. 베이스라인 rate
> 측정 + 모든 stage 를 NVTX range 로 감쌈 → 이후 phase 의 효과를 Nsight 로
> 명확히 측정 가능.

- [ ] `configs/resnet18_level_a_smoke.yaml`
  - shard_dir 경로 → `94_production_102k_x200`
  - aux_labels 경로 → 동일
  - input_size 192 → **128** (corpus 가 128 로 생성됐고, decode 시 upscale 은 무의미한 비용)
  - batch_size 128 → 256 (12.9 GB VRAM 여유 측정 후 결정)
- [ ] **NVTX 계측** — `train_loop.py` 의 train_one_epoch 안 각 stage 에 `torch.cuda.nvtx.range_push("name") / range_pop()` 추가:
  - `data_load`: `for x, y in loader` 의 yield 단계
  - `h2d`: `.to(device, non_blocking=True)` + `aux_table.get_aux(y)`
  - `gpu_transform`: float/resize/normalize
  - `forward`: `model(x)` 호출
  - `loss`: `compute_multitask_loss(...)`
  - `backward`: `scaler.scale(loss).backward()`
  - `optim`: `scaler.step(optimizer); scaler.update()`
  - `evaluate.*`: 동일 패턴
- [ ] smoke 1 epoch 실행 → step rate (img/s), GPU util, VRAM peak 기록
- [ ] **Nsight Systems 1회 캡처** — `nsys profile -o tg0_baseline ...` 으로 baseline timeline 저장. 이후 phase 의 비교 기준
- [ ] **합격 기준**: 5-head loss 다 감소, char/top1 > 1% (random > 1/98169 = 0.001%), eval 안 죽음
- [ ] 진행률 / sysmon 로그 v2 패턴으로 갖춰져 있는지 확인 (없으면 추가)

산출물: `train_engine_v3/out/00_smoke/smoke_result.json` + `smoke.log` + `tg0_baseline.nsys-rep`

### Phase TG-1 — PyTorch Native Free Wins  ✅ 완료

> **결과**: 1.5-2× 기대했으나 실측 +20.9%. channels_last / bf16 / compile 전부
> 부정 결과. 자세한 내용 [doc/23](23_PHASE_TG1_RESULTS.md). 핵심: batch size
> 키우고 cudnn.benchmark 켜는 두 줄이 거의 모든 win.

> **목표** (원본): custom kernel 작성 없이 1.5-2× 받기.

각각 적용 후 baseline 대비 step rate 측정.

- [ ] **channels_last memory format**
  - `model = model.to(memory_format=torch.channels_last)`
  - `gpu_transform` 출력에 `.contiguous(memory_format=torch.channels_last)` 추가
  - 합격: rate +20-50%, accuracy 동등
- [ ] **cudnn.benchmark = True**
  - `torch.backends.cudnn.benchmark = True` (입력 크기 일정할 때만 안전)
  - 합격: rate +0-5% (warmup 후 측정)
- [ ] **bf16 AMP**
  - GradScaler 불필요, autocast(dtype=torch.bfloat16) 로 교체
  - 합격: rate 동등, val loss 안정 (fp16 grad scale issue 제거)
- [ ] **torch.compile**
  - `model = torch.compile(model, mode="reduce-overhead")` 또는 `"max-autotune"`
  - 첫 step warm-up 길어짐 — log 에 명시
  - 합격: 두번째 epoch 부터 +30-100%
- [ ] **batch size sweep**
  - 256, 384, 512 시도 — VRAM 와 rate trade-off
  - 합격: VRAM peak < 11 GB (안전마진), rate maximize

산출물: phase_TG1_results.md — 각 옵션의 rate/VRAM 표.

### Phase TG-2 — Custom CUDA: Fused Multi-Head Loss + Aux Gather

> **목표**: train_loop.py 의 (7) AuxTable gather + (8) loss 5개를 1-2 kernel
> 로 융합. 이론 win 2-4% of step.

**현재 코드** ([train_loop.py:65-92](../train_engine_v3/modules/train_loop.py#L65-L92)):
```python
l_char = F.cross_entropy(logits["char"], y_char, label_smoothing=0.1)
l_rad  = _masked_ce(logits["radical"], aux.radical, aux.valid[:,0])
l_tot  = _masked_smooth_l1(logits["total_strokes"], aux.total, aux.valid[:,1])
l_res  = _masked_smooth_l1(logits["residual_strokes"], aux.residual, aux.valid[:,2])
l_idc  = _masked_ce(logits["ids_top_idc"], aux.idc, aux.valid[:,3])
total = w_char*l_char + w_rad*l_rad + w_tot*l_tot + w_res*l_res + w_idc*l_idc
```

5 kernel launch + 5 reduction. 각각 작아서 launch overhead 가 무시 못 함.

**custom kernel 설계**:
```
fused_aux_loss<<<grid, block>>>(
    logits_radical_ptr, logits_total_ptr, logits_residual_ptr, logits_idc_ptr,
    aux_table_ptr,                    // (n_class, 4) packed
    char_labels_ptr,                  // (B,) — 한 번만 인덱싱
    valid_mask_ptr,                   // (n_class, 4)
    weights_ptr,                      // 4 floats
    output_loss_ptr                   // (1,) atomicAdd target
)
```

- 각 thread block 이 batch 의 일부 sample 처리
- shared memory: (block_size, 4) aux 라벨 (gather 결과 캐시)
- per-sample 4 loss 계산 후 weighted-sum, block reduction → atomicAdd
- backward 는 별도 kernel — gradient 도 마찬가지로 fuse

**적용가능 기법** (batched-lenet-cuda repo):
- **L6 reduction**: per-sample loss → block reduction → atomicAdd 으로 final loss
- **L3 tiled**: shared memory 로 aux gather 결과 block 안에서 재사용 (4-tuple radical/total/residual/idc)
- **Project kernel fusion**: 5→1 kernel
- **Triton 대안**: raw CUDA 작성보다 Triton (`@triton.jit`) 으로 fused kernel 작성하면 코드량 1/5, 거의 동등 성능. raw CUDA 는 학습 목적, Triton 은 production 목적. 두 가지 다 작성해서 측정·비교 가치 있음 (synth 의 photometric 과 동일 교훈 — 자동화 도구가 hand-tuned 와 비길 때도 많음)

**합격 기준**:
- step rate 향상 측정 (이론 +2-4%)
- output loss / gradient 가 PyTorch reference 와 |diff| < 1e-4
- backward 가 register accumulator 로 grad 정확도 유지

**Honest expectation**: char head CE 가 워낙 dominant (98169-way softmax) 라
fused 가 수치적으로 합쳐줘도 char 부분은 그대로 cross_entropy 에 맡기고 aux
4개 만 fuse 하는 게 실용적. char 까지 직접 fused 하면 softmax 구현이 위험
(numerical stability 처리 직접 해야).

→ **fused_aux_only** 로 범위 한정.

### Phase TG-3 — Custom CUDA: Data Pipeline (가장 큰 win 후보)

> **목표**: shard NPZ → batch tensor pipeline 의 CPU side 비중을 GPU 로 옮김.
> 이론 win 7-10% of step.

**현재 ([shard_dataset.py:116-129](../train_engine_v3/modules/shard_dataset.py#L116-L129))**:
- `_iter_shard`: per-sample Python yield, `np.ascontiguousarray + transpose + from_numpy`
- DataLoader workers 가 CPU 에서 batch 수집 → pin_memory → H2D
- 5000 샘플 shard 마다 numpy 디코드 5000번, transpose 5000번

**redesign**:
```
새 BatchedShardLoader:
  - prefetch thread: NPZ unpickle → uint8 numpy (N, 128, 128, 3) 한번에
  - GPU 로 한 번에 H2D (one big copy, async, pinned host buffer)
  - GPU kernel: uint8 NHWC → float NCHW + normalize 통합 1 launch
  - random shuffle 은 GPU index permutation (cuRAND 또는 단순 permute)
  - batch slice: 그냥 pointer offset
```

**적용 기법** (batched-lenet-cuda):
- **Project streams**: compute_stream / copy_stream 분리 + cudaEvent 동기화
- **더블 버퍼 pinned host** (synth OPT-Final 과 동일 패턴)
- **Kernel fusion**: uint8 NHWC → float NCHW + normalize 1 kernel (synth 의 raster_kernel_v2 와 비슷한 register tiling 적용 가능)
- **L5 privatized atomics** (옵션): 배치 내 class-frequency 카운트 — 학습 중 클래스 균형 모니터링 (현재 안 하지만 추가 가능)

**합격 기준**:
- DataLoader 의 CPU bottleneck 해소: nsight 에서 GPU idle wait < 5% of step
- VRAM 추가 사용 < 1 GB (한 shard 만큼 prefetch)
- shard order seed 재현성 보장

**Risk**: 현 PyTorch DataLoader 의 IterableDataset / num_workers 모델과 어떻게
어울릴지. 한 가지 옵션: workers=0 + 직접 prefetch thread 가 GPU 로 직접 push.
시너지 측정 필수 (TG-1 channels_last + compile 와 결합 시 추가 win 0~5%).

### Phase TG-4 — 측정 / 정리

- [ ] 각 phase 의 단일 변수 rate 측정표 작성 → phase_TG_summary.md
- [ ] nsight systems profile 1-2 epoch → 남은 hot spot 진단
- [ ] doc/22 § "13" 같은 final summary table

---

## 3. 측정 표준

모든 phase 동일 baseline 으로 비교:
- corpus: 94_production_102k_x200 (98,169 class)
- shards: 32 (160k samples) — quick run, cache stable
- batch: 256 (또는 TG-1 sweep 결과)
- input_size: 128
- AMP: bf16
- epochs: 2 (1st = warmup, 2nd = 측정)
- log every 50 steps

기록할 metric:
- `rate (img/s)` — 2nd epoch steady state
- `gpu_util %` — 2nd epoch median
- `vram_torch peak (GB)`, `vram_dev peak (GB)`
- `eval/char_top1` — sanity 만 (정확도 측정 아님, 같은 학습 길이에서 동등 ±1%)
- `walltime per epoch (s)`

---

## 4. 예상 Negative Findings (선언)

- **char head fused**: numerical stability 깨짐. 안 함.
- **conv layer 직접 작성**: cuDNN 보다 빠른 conv 못 만듦. 안 함.
- **resnet block 직접 작성**: 동일.
- **softmax 직접 fused**: torch.cross_entropy 가 이미 fused. 안 함.
- **H2D 의 torch.compile fusion**: input 이 dynamic shape 일 때 fallback. 측정 후 결정.

이 4개 항목은 doc/20 의 "fused photometric" 처럼 -13% loss 가 될 가능성 높음.
처음부터 시도하지 않는다. (synth 때 학습한 교훈.)

---

## 5. 진입 조건 + 출구 조건

각 phase 시작 전:
- 이전 phase 의 합격 기준 충족
- corpus / aux_labels 변경 없음 (TG-0 이후 frozen)

각 phase 출구:
- rate / VRAM / accuracy 기록 commit
- 부정적 결과여도 기록 (synth 때 streams 0%, photometric -13% 처럼)

---

## 6. 한 줄 요약

> **Conv 부는 cuDNN 에 맡기고, 5-head loss + aux gather 와 data pipeline 두
> 곳에 custom CUDA 를 집중. PyTorch native 기법으로 1.5-2× 받고, custom 으로
> 추가 1.06-1.13× 받아 총 1.6-2.3× 가속을 목표.**

# Phase TG-1 결과 — PyTorch native free-win 측정

작성: 2026-04-26.  계획: [doc/22 §TG-1](22_TRAIN_GPU_OPTIMIZATION_PLAN.md).
대상 corpus: `synth_engine_v3/out/94_production_102k_x200/` (98,169 class).

각 변경을 **단일 변수** 로 측정 → win 인지 loss 인지 정직하게 분류. doc/22 의
모든 win 추정치는 hypothesis 였고, 이 문서는 실측 결과.

---

## 1. 측정 환경

- 하드웨어: RTX 4080 Laptop (12.9 GB)
- 모델: ResNet-18 multi-head (61.65 M params)
- 입력: 128×128 RGB
- AMP: fp16 (default), bf16 (variant)
- corpus: 32 shards × 5000 = 160k samples (val 0.0625), 2 epoch
- rate 계산: epoch 2 (steady state) wall time / train sample 수
- venv: `.venv/Scripts/python.exe` (Windows native, NTFS direct)
- 비교 baseline: TG-0 Windows (3,373 img/s)

---

## 2. 결과표

| run | variant | bs | platform | steady rate (img/s) | vs Win baseline | VRAM peak | loss e2 |
|---|---|---:|---|---:|---:|---:|---:|
| `01_tg0_baseline` | **baseline (256, fp16)** | 256 | win | **3,373** | **0%** | 2.56 GB | 2.43 |
| `02_tg1_bs384` | bs=384 | 384 | win | 3,749 | **+11.2%** | 3.20 GB | 4.68 |
| `03_tg1_bs512` | bs=512 | 512 | win | 3,822 | +13.3% | 3.84 GB | 6.61 |
| `04_tg1_bs640` | **bs=640** | 640 | win | **3,989** | **+18.3%** | 4.68 GB | 7.42 |
| `05_tg1_cudnn` | cudnn.benchmark=True | 256 | win | 3,373 | 0% | 2.43 GB | 2.61 |
| `06_tg1_channels_last` | channels_last | 256 | win | 1,206 | **−64.2%** ❌ | 2.56 GB | 2.23 |
| `07_tg1_bf16` | bf16 AMP | 256 | win | 2,168 | **−35.7%** ❌ | 3.85 GB | 2.26 |
| `09_tg1_combined_minimal` | **bs=640 + cudnn** | 640 | win | **4,078** | **+20.9%** ★ | 4.96 GB | 7.62 |
| `10_tg1_combined_chlast` | bs=640 + cudnn + chlast | 640 | win | 1,220 | −63.8% ❌ | 4.50 GB | 7.31 |
| `08_tg1_compile` | torch.compile (reduce-overhead) | 256 | **WSL** | 3,088 | −8.4% ✱ | 2.79 GB | 2.37 |
| `01_tg0_baseline_wsl` | baseline (참고) | 256 | **WSL** | 2,668 | −20.9% (WSL 비용) | 3.10 GB | 2.47 |

★ = TG-1 best.  ❌ = 부정적 결과 (적용 안 함).  ✱ = WSL/triton 한정.

> **loss_e2 비교 주의**: 더 큰 batch → 같은 epoch 안에서 step 수 감소 → loss 덜
> 수렴. TG-1 측정 목적은 **rate 비교** 이므로 loss 절대값 비교 무의미. 본격
> 학습에선 lr scaling 으로 보정.

---

## 3. 분석

### 3.1 Win — Batch sweep (+18.3% @ bs=640)

batch size 가 가장 cheap 한 win. RTX 4080 Laptop @ 128px, 12.9 GB VRAM 기준
**bs=640 까지 안전** (peak 4.68 GB, 36% utilization). bs 더 키울 여지 있음
(bs=896 / 1024 시도 가치) 단, augment 없는 큰 배치는 학습 dynamics 영향 → 본
학습엔 lr-batch 스케일링 + step 수 보정 필요.

### 3.2 Win — Batch + cudnn 조합 (+20.9% @ combined_minimal)

cudnn.benchmark 단독은 0% — cuDNN 의 default heuristic 이 이 layer 셋엔 이미
optimal. 하지만 **bs=640 과 결합** 시 +2.6% 추가 (bs=640 단독 3,989 → 결합
4,078). batch 가 커지면 algorithm-selection space 가 넓어져 benchmark 가 더
나은 kernel 찾을 여지 생김.

### 3.3 Loss — channels_last 단독 −64.2% / 결합 −63.8% ❌

가장 의외의 결과. 일반적으로 Ada Tensor Core 는 NHWC 선호로 알려져 있는데
이 워크로드에선 정반대.

**가능한 원인:**
- 입력이 작음 (128px): NHWC 변환 overhead 가 compute 대비 크게 보임
- ResNet-18 은 layer 가 적어 layout transition 비용 분산 안 됨
- AMP fp16 + channels_last 조합에서 cuDNN 이 비효율 path 잡음 (heuristic 미스)
- cudnn.benchmark + channels_last 결합 시도 (`10_combined_chlast`) 도 −63.8% — algorithm 선택 늘어나도 여전히 NHWC 가 느림

**결정**: TG-1 에서 채택 안 함. 192px / 256px 큰 입력에서 재시험할 만 (T4 단계).

### 3.4 Loss — bf16 AMP −35.7% ❌

또 하나 의외. fp16 과 동등 속도 기대했는데 bf16 이 35% 느림.

**가능한 원인:**
- Ada Laptop 의 fp16 path 가 bf16 보다 더 잘 최적화돼 있음 (consumer GPU 의
  특성 — datacenter Hopper 와 다름)
- GradScaler 제거 (bf16 은 dynamic range 충분해서 안 씀) 가 어떤 fast path
  를 끊었을 가능성
- cuDNN heuristic 이 bf16 conv 에 덜 최적화된 algorithm 선택

**결정**: TG-1 에서 채택 안 함. 수치 안정성 이슈 발생 시에만 재고려.

### 3.5 ✱ — torch.compile WSL 한정 (+15.7% over WSL base, but Windows-overall 비교 안 됨)

Windows .venv 는 triton 미설치 → compile 불가. WSL 에서 측정:
- WSL baseline: 2,668 img/s
- WSL + compile: 3,088 img/s (**+15.7%** WSL 내 비교)
- 단 Windows baseline (3,373) 자체보다 느림

**결론**: WSL 환경 자체가 Windows 보다 −20.9% 패널티. WSL+compile 의 +15.7%
도 그 패널티 일부만 회복. 즉 **production 학습은 Windows 가 유리**, compile
은 WSL-only 옵션. 단 long-run 에서 cuDNN benchmark / inductor caching 효과
누적되면 다를 수 있음 (smoke 2-epoch 측정의 한계).

향후: Windows 에서 triton 설치 가능해지면 (PyTorch 의 pip 의존성에 들어오면)
재측정.

### 3.6 추가 발견 — Platform overhead

WSL → /mnt/d 의 9P bridge 가 데이터 로드 hitch 를 키움:
- WSL baseline: 2,668 img/s
- Windows baseline: 3,373 img/s
- **WSL 패널티 = −20.9%**

이 차이의 정체는 [doc/22 §TG-3 (data pipeline)](22_TRAIN_GPU_OPTIMIZATION_PLAN.md) 에서 다룰 shard NPZ
read latency. Windows 가 NTFS native 로 직접 읽어서 hitch 가 약함. WSL 은 9P
protocol overhead. 이는 TG-3 의 data-pipeline custom CUDA 로 양 platform 모두
개선 가능 — production 환경 무관.

---

## 4. TG-1 합격 winners

| 적용할 변경 | 효과 | 위험 |
|---|---|---|
| `batch_size: 640` (또는 896 시도) | +18.3% | bs↑ 시 lr 스케일링 필요 (linear scaling rule) |
| `cudnn.benchmark: True` | +0~+2.6% | input shape 일정 가정. T3 학습은 일정 |

→ **`bs=640 + cudnn = +20.9% 베이스라인 향상`**.

## 5. TG-1 불합격 (적용 안 함)

| 변경 | 결과 | 사유 |
|---|---|---|
| channels_last | −64.2% | NHWC 가 이 워크로드에서 felonious slow. 128px 입력의 한계 가능성 |
| bf16 AMP | −35.7% | Ada Laptop fp16 path 가 더 빠름 |
| torch.compile (Windows) | n/a | triton 미지원 |
| torch.compile (WSL) | +15.7% local but Windows-overall 못 따라감 | platform mismatch |

doc/22 §4 의 "Honest Negative Findings" 섹션에 추가 기록 가치 있음 — synth
의 fused photometric (−13%) 과 같은 결의 학습된 부정 결과.

---

## 6. 다음 단계 제안

**TG-2 (fused multi-head loss)** 와 **TG-3 (data pipeline)** 두 트랙 중:

- TG-2: 이론 win 2-4%. small. 구현 위험 ⊃ kernel correctness verify
- TG-3: WSL 데이터에서 보였던 shard hitch (35% 변동) 은 Windows 에선 약함.
  Windows 측정에서도 hitch 패턴이 잔존하는지 nsys 캡처 필요. 잔존하면 그대로
  진행, 사라졌으면 TG-3 우선순위 하향.

**추천 진입 순서**:
1. **현 best (`bs=640 + cudnn`) 로 nsys profile** — 잔존 bottleneck 확인
2. nsys 결과에 따라 TG-2 / TG-3 결정
3. Combined winner 로 T3 baseline 학습 진입 (10,932 class 로 v2 비교 재현)

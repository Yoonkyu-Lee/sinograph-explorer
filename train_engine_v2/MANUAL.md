# train_engine_v2 — Manual

v1 의 확장. PNG 기반 `CorpusDataset` (v1 에서 완성) 그대로 유지 + **tensor_shard 경로 추가**. v3 generator 의 `--output-format tensor_shard` 산출물을 직접 로드.

## 개요

Stage 2 (glyph → codepoint 분류기) 학습 엔진. Stage 1 산출물을 읽어 ResNet-18 을 학습하고 ckpt / ONNX 을 낸다. 상세 설계는 `doc/12_STAGE2_TRAINING_PLAN.md` + `doc/13_AUGMENT_IO_COEVOLUTION_PLAN.md`.

## v1 대비 변경점

| 항목 | v1 | **v2** |
|---|---|---|
| 데이터 포맷 | PNG × 5.45M + `corpus_manifest.jsonl` | **+ `.npz` tensor shards** (PNG 도 지원) |
| `modules/dataset.py` | `CorpusDataset` (PIL decode) | 그대로 (v1 호환) |
| `modules/shard_dataset.py` | 없음 | **NEW** — `TensorShardDataset` (IterableDataset) |
| DataLoader shuffle | `shuffle=True` | IterableDataset 모드: shard level + buffer shuffle |
| Best-ckpt tracking | ckpt per epoch 만 | **+ `best.pth` (val top-1 기준) + `best.json`** |
| config 형식 | manifest/image_root 필수 | `data.format` 분기 (`png` / `tensor_shard`) |
| 예상 throughput | M4 steady 3,185 img/s (hot), ~888 (cold) | shard path ~6k-10k (hot), ~2-3k (cold) |

## 환경 (v1 과 동일)

## 환경

- Windows 11, GPU **NVIDIA RTX 4080 Laptop GPU, 12 GB 전용 VRAM** (shared 17.9 GB 별도)
- System RAM 약 34 GB
- Python venv: `.venv/Scripts/python.exe`
- torch 2.11.0+cu128, torchvision 0.26.0
- 추가 설치: `pyyaml`, `matplotlib`, `onnx`, `onnxruntime`, `nvidia-ml-py`, `psutil`
- Data: `synth_engine_v3/out/42_production_mobile/corpus_manifest.jsonl` (1.2 GB, 5,450,000 rows, 10,932 classes, T1 mobile 문자)

## 스크립트

| path | 역할 |
|---|---|
| `scripts/00_cpu_smoke.py` | CPU PNG 경로 smoke (v1 과 동일) |
| `scripts/00_cpu_smoke_shards.py` | **NEW** — CPU tensor_shard 경로 smoke (fake .npz 3 개 생성 후 학습 루프 통과) |
| `scripts/20_train.py` | 메인 학습 CLI. yaml + 오버라이드 + `--data-format` 지원 |
| `scripts/21_eval.py` | ckpt + manifest 로 top-k / family-aware 평가 |
| `scripts/22_export_onnx.py` | ckpt → ONNX (opset 17, dynamo=False) |
| `modules/dataset.py` | v1 그대로: `CorpusDataset` (PNG + PIL decode) |
| `modules/shard_dataset.py` | **NEW** — `TensorShardDataset` (IterableDataset, .npz 직접 로드) |
| `modules/model.py` | `build_model("resnet18", num_classes)` |
| `modules/train_loop.py` | `train_one_epoch`, `evaluate` (AMP, sysmon, gpu_transform 훅) |
| `modules/sysmon.py` | GPU util / VRAM / RSS |
| `modules/family_eval.py` | canonical_v2 SQLite family-aware metric |
| `modules/utils.py` | `Tee` 로그, checkpointing, `plot_curves` |

## data.format 분기

`cfg["data"]["format"]` 또는 `--data-format` CLI:

### `format: tensor_shard` (v2 primary)
```yaml
data:
  format: tensor_shard
  shard_dir: .../80_production_v3r_shard256
  val_ratio: 0.1
```
- shard_dir 에서 `shard-*.npz` + `class_index.json` 자동 탐지
- Train/val split 은 **shard 단위** (rough 10% → val_ratio 에 따라)
- IterableDataset: worker 간 shard disjoint 분배 + within-shard shuffle buffer
- DataLoader 가 자동 `set_epoch` 으로 shuffle seed 재조정
- GPU 에서 `uint8 → float/255 → (shard size != model size 면 resize) → normalize`

### `format: png` (v1 backward compat)
```yaml
data:
  format: png
  manifest: .../corpus_manifest.jsonl
  image_root: .../42_production_mobile
  val_ratio: 0.1
  decode: pil              # or tvio
  gpu_transforms: true     # v1 M4 winner setting
```
v1 과 동일. 기존 ckpt/config 그대로 호환.

## CLI 오버라이드 (20_train.py)

yaml 수정 없이 tuning 반복:
```
--run-tag <str>             # out_dir 에 "__<tag>" 접미사 — 결과 격리
--epochs N                  # config.train.epochs
--batch-size N
--num-workers N
--max-classes N             # (png 모드만) config.data.max_classes
--input-size N              # config.model.input_size
--amp on|off
--prefetch-factor N
--decode pil|tvio           # (png 모드만) config.data.decode
--gpu-transforms on|off     # (png 모드만) v1 M4 M4 윈너
--data-format png|tensor_shard   # config.data.format
--seed N
```

## 기본 실행 (예시)

```bash
# v3 realistic shard 기반 mini pilot (1k class proxy, 5 epoch)
python train_engine_v2/scripts/20_train.py \
  --config train_engine_v2/configs/resnet18_t1_v3r_mini.yaml

# Production 10,932 classes × 20 epoch
python train_engine_v2/scripts/20_train.py \
  --config train_engine_v2/configs/resnet18_t1_v3r_full.yaml

# v1 호환 PNG 경로 (이미 있는 42_production_mobile 재학습시)
python train_engine_v2/scripts/20_train.py \
  --config train_engine_v1/configs/resnet18_t1_pilot.yaml --data-format png
```

## Best-ckpt tracking

학습 중 매 epoch 마다 `ckpt_epoch_NN.pth` 저장 + **val top-1 최고** 한 epoch 의 ckpt 를 별도 복사:
- `out_dir/best.pth` — 최고 ckpt
- `out_dir/best.json` — `{"best_top1": ..., "best_epoch": ..., "final_epoch": ...}`

Overfitting 발생 (val top-1 감소) 시 `best.pth` 가 production inference 에 쓸 것.

`30_predict.py` 실행시 `--ckpt .../best.pth` 권장.

## 실시간 로그 관찰

```powershell
# PowerShell, 특정 trial
Get-Content "d:/Library/.../train_engine_v1/out/01_pilot_mini__<TAG>/run.log" -Wait -Tail 50

# 최신 수정된 run.log 자동
Get-ChildItem "d:/.../train_engine_v1/out/01_pilot_mini__*/run.log" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName |
  ForEach-Object { Get-Content $_ -Wait -Tail 50 }
```

## 로그 판독

```
[22:37:52] step 50/351  loss=3.6276  (422.2 img/s, t=15s, eta=91s)  gpu=20% (min 1, max 95, n=74) vram_torch=0.18/0.98GB vram_dev=3.27/12.0GB rss=8.93GB sys=26.0/34GB
```

- `[HH:MM:SS]` wall clock — 라이브 tail 에서 stale vs 진행 중 즉시 구분
- `(X img/s, t=Ns, eta=Ms)` = cumulative rate since epoch start + 경과초 + 남은 시간 추정. warmup 때문에 초반 낮음. **steady-state 는 마지막 창 또는 per-window 계산으로 판단**
- `gpu=X% (min A, max B, n=N)` = **이 창 (약 4s) 에 200ms 간격으로 N회 샘플** 평균/최소/최대
- `vram_torch=A/B` = torch 가 보유한 tensor 메모리 (alloc / reserved). **순수 모델·activation·gradient 만**
- `vram_dev=C/D` = nvml 이 보는 device-wide VRAM (사용량/전체). **Windows 작업관리자의 "전용 GPU 메모리" 와 일치**. vram_torch 에 CUDA 컨텍스트 (~1 GB) + cuDNN workspace (~500 MB) + 타 프로세스 가 더해짐
- `rss=X` = 파이썬 프로세스 메모리. manifest + class_index 로드로 초반 8~10 GB. **에폭 간 증가하면 leak 의심**
- `sys=A/B` = 시스템 RAM 사용/전체. 상승해도 `rss` 가 일정하면 OS 파일 캐시 (정상)

### 두 VRAM 수치 선택 기준
- **누수 탐지 / batch 확장 여유**: `vram_torch` (깨끗한 tensor 만)
- **GPU 포화도 / 작업관리자 대조**: `vram_dev`

---

# Throughput Tuning (S2-A)

**목적**: Windows + RTX 4080 16 GB 에서 ResNet-18 @ 128² / T1-mobile 100-class pilot 의 최적 `num_workers`, `batch_size`, `prefetch_factor` 찾기.

**공통 pilot 설정**: 100 class × 500 = 50k train + 5k val (10% split), 1 epoch per trial, AMP on, SGD + cosine + label_smoothing 0.1.

**측정 3 축**:
1. **Total wall_s** — 1 epoch 전체 시간 (warmup 포함). 파일럿 개발 사이클에 중요
2. **Steady-state img/s** — warmup 후 창별 rate. 20-epoch production 에 유효 (persistent_workers=True 로 spawn 1회만)
3. **Peak GPU util / VRAM** — 여유 판별

## Trial 요약 표

| # | tag | w | bs | decode | wall_s | steady img/s | mean GPU% | peak VRAM_torch GB | peak RSS GB | val top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | baseline | 4 | 128 | pil | **53.1** | ~1600 | 25~50 | 0.68 | 8.94 | 0.998 |
| T2 | w8 | 8 | 128 | pil | 80.7 | ~2250 | 30~50 | 0.68 | 9.01 | 0.999 |
| T3 | bs256 | 4 | 256 | pil | 91.1 | ~800 | 12~20 | 1.21 | 9.11 | 0.997 |
| **T4** | **bs256_w8** | **8** | **256** | **pil** | **69.3** | **~2300** | 15~49 | 1.21 | 9.71 | 0.999 |
| T5 | bs512_w8 | 8 | 512 | pil | 86.8 | ~1075 | ~19 | 2.27 | 10.13 | 0.955 |
| T6 | bs256_w8_tvio | 8 | 256 | tvio | 89.0 | ~1115 | 14~19 | 1.21 | 9.25 | 0.998 |
| T7 | bs256_w12 | 12 | 256 | pil | 100.1 | ~2311 | 16~62 | 1.21 | 9.99 | 0.999 |
| T8 | bs256_w8_gputx | 8 | 256 | tvio | 81.1 | ~1440 | 19~58 | 1.21 | 9.18 | 0.997 |
| T9 | (T8 재실행, 의도치 않음) | 8 | 256 | tvio | 71.4 | ~3200 | 15~38 | 1.21 | 8.77 | 0.998 |

## Trial 상세 기록

### T1 — `baseline`
- config: pilot.yaml 그대로 (workers 4, batch 128, prefetch default)
- warmup end: 약 t=15s (step 50)
- steady-state window rate (step 50~350, 창당 50 step):
  ```
  50→100: 1600 img/s, 100→150: 2133, 150→200: 1600, 200→250: 1600, 250→300: 1600, 300→350: 1600
  ```
- **train 부분만** ≈ 38s, eval ≈ 15s (5000 val @ workers 4)
- GPU util 평균 25~50% — 작은 batch 때문에 kernel launch + Python overhead 가 큼
- 진단: batch 를 올리면 GPU util ↑ 유력

### T2 — `w8`
- config: baseline + `--num-workers 8`
- warmup end: 약 t=33s (step 100, T1 대비 +18s)
- steady-state (마지막 창 300→350): 3200 img/s, 전반 steady 평균은 1500~2200
- wall_s 80.7 > T1 53.1 인데 **steady 는 빠름** — spawn 오버헤드가 지배
- 20-epoch 환산 (first_epoch + 19 × steady_epoch) 로 예상 교차:
  - T1: 53 + 19×30 ≈ 623 s
  - T2: 80 + 19×26 ≈ 574 s (만약 steady epoch 이 T1 대비 15% 빠르면)
- 결론: **steady-state 가 정식 tuning 기준**. 단일 epoch wall time 은 잘못된 신호
- 다음 단계: batch 를 올려 GPU util 근본 상승 확인

### T3 — `bs256`
- config: T1 + `--batch-size 256`
- 반전 관찰: wall_s **91s** (T1 보다 +72%). steady 창 rate **~800 img/s** (T1 의 절반)
- mean GPU util **12~20%, min=1%** — GPU 가 거의 놀고 있음
- 원인: batch 256 의 I/O 요구량을 **workers 4 가 못 쫓아감**. GPU 는 batch 를 빨리 처리 후 다음 batch 기다림
- 결론: **batch ↑ 는 workers ↑ 와 동시에** 올려야 유효. batch 단독 실험은 misleading

### T4 — `bs256_w8` ⭐ **승자 (PIL 범위 내)**
- config: T1 + `--batch-size 256 --num-workers 8`
- warmup end: t=28s (8 workers spawn 시간), 이후 steady
- steady 창 rate: step 50→100 **2560 img/s**, step 100→150 **2133 img/s** → 평균 **~2300 img/s**
- T1 대비 **+44% throughput**, T3 대비 **2.9× 개선** (I/O 맞춤 확보)
- mean GPU util 15~49%, peak 98% (순간) — 개선은 됐지만 **여전히 steady 50% 미만**
- 진단: I/O 를 workers 가 맞춘 뒤에도 GPU 에 여유 많음 → **이번엔 CPU-side 디코딩 (PIL PNG) 이 병목** 후보
- peak RSS 9.71 GB (T1+0.77GB, 8 workers 추가 overhead)

### T5 — `bs512_w8`
- config: T4 + `--batch-size 512`
- wall_s 86.8s, steps 87 (log_every=50 이 한 번만 발화 → steady 창 데이터 부족)
- 추정 steady ~1075 img/s (T4 의 절반 이하로 후퇴)
- peak VRAM_torch 2.27 GB / reserved 3.55 GB — 여전히 여유. VRAM 이 bottleneck 아님
- val top-1 **0.955** (T1~T4 의 0.997+ 대비 낮음) — 이건 1 epoch 에 대한 update 수 감소 때문, throughput 판단엔 무관
- 결론: **workers 8 도 bs=512 의 I/O 요구량 못 맞춤**. bs256 이 한계. 또는 workers 12+ 가 필요하겠지만 Windows spawn 오버헤드 상승 trade-off

### T6 — `bs256_w8_tvio`
- config: T4 동일 + `--decode tvio` (torchvision.io.read_image)
- 결과: **예상 반전**. wall_s 89s (T4 69s 보다 +29%), steady 창 ~1070 img/s (T4 의 절반)
- mean GPU util 14~19% — T4 (15~49%) 대비 오히려 나빠짐
- 원인: torchvision.io 의 decode 자체는 빨라도, **uint8 tensor 위에서 `transforms.Resize(128, antialias=True)` 가 PIL SIMD 경로보다 느림**. Decode + resize net 합산에서 PIL 이 승
- 결론: **이 워크로드 (256→128 resize) 에서는 PIL 가 최적**. 만약 원본이 훨씬 커서 resize 비율이 크면 다른 결과 가능하지만 우리는 이미 256x256 으로 생성함

### T7 — `bs256_w12`
- config: T4 + `--num-workers 12`
- wall_s **100.1s** (T4 69s 보다 +45%)
- steady 창 rate 양분:
  - step 50→100: 1422 img/s (아직 warmup 중)
  - step 100→150: **3200 img/s**, gpu=62% (마지막 창 burst)
- **전체 steady 평균 ~2311 img/s** — T4 와 거의 동일
- eval 오버헤드 큼: step 150 → 완료 까지 ~42s, 대부분 val dataloader 워커 재스폰
- **발견된 버그**: `dl_val` 에 `persistent_workers=False` 를 강제하고 있었음. train 과 같은 dl_kwargs 그대로 쓰도록 수정 완료
- 결론: w12 의 spawn 은 크지만 steady throughput 은 w8 와 본질적으로 같음. production (persistent_workers=True, spawn 1회) 에서도 **w8 로 충분**

### T8 — `bs256_w8_gputx` (gpu_transforms=on, decode=tvio)
- 목적: CPU decode/IPC 부담 경감 실험. Dataset 이 uint8 tensor 만 리턴 → GPU 가 resize+normalize
- config 변경: float32 IPC (50 MB/batch) → uint8 IPC (12.5 MB/batch), CPU transform pipeline 대폭 축소
- 결과: wall_s 81.1s, steady window ~1440 img/s, mean GPU util 19~58%
- **마지막 창에서 gpu=58%** 로 올라감 (T4 마지막 49% 대비 +9%p). GPU 포화도 자체는 개선
- 그러나 **전체 throughput 은 T4 보다 약 40% 하락** — tvio 의 decode 가 PIL 보다 느린 이유 (T6 와 같은 관찰). uint8 IPC 이득 < tvio decode 손실
- VRAM_reserved 1.51 GB (T4 1.71 보다 조금 낮음 — uint8 IPC), RSS 9.18 GB (낮음, CPU transform 안 해서)
- 결론: gpu_transforms 는 이 워크로드에 이득 없음. CPU transform 비용은 이미 작고, decode 가 진짜 병목이라는 걸 재확인

### T9 — 원래 "bs256_w8_pilgpu" 의도, 실제로는 T8 재실행 (버그)
- 의도: PIL decode (빠름) + uint8 IPC + GPU normalize 하이브리드
- 버그: `gpu_transforms=on` 분기에서 `decode="tvio"` 로 강제 덮어쓰기 하는 코드가 남아있었음. `--decode pil` 가 무시됨
- 결과: loss 1.7612067495073591 이 T8 과 소수점까지 일치 → T8 와 같은 config 재실행이었음을 확인
- **예상치 못한 발견**: 같은 config 인데 wall_s T8 81s → T9 71s (−12%). **OS NTFS 가 T8 에서 읽은 45k PNG 를 캐시** → T9 는 hot cache 로 진행
- **이 관찰이 단일-epoch trial 의 한계를 직접 증명**. 같은 설정에서도 10%+ noise. T1~T7 의 비교에도 캐시 효과가 섞여있을 가능성
- 이후: 버그 수정 완료 (강제 덮어쓰기 제거). 추가 pilot isolation trial 대신 **Mini 실전 학습 (multi-epoch) 에서 진짜 steady-state 관찰**로 넘어감

## ⚠️ 단일-epoch 결론 (T1~T9) 은 부정확. 진짜 결과는 아래 Multi-epoch 섹션.

T8 → T9 재실행 비교에서 같은 config 에서도 **10%+ wall_s 차이** 가 NTFS 캐시 때문. 1-epoch trial 의 한계 인지하고 **M-series (multi-epoch)** 로 재검증.

---

# Multi-epoch 검증 (M1~M5)

각 trial 3 epoch 으로 epoch 1 (cold + spawn) 과 epoch 2/3 (steady, hot cache + persistent workers) 를 직접 비교.

## M-series 종합 표

| | M5 (w4 bs128) | M1 (w8 bs256 pil) | M2 (w12 bs256 pil) | M3 (w8 bs256 tvio gputx) | **M4 (w8 bs256 pil + gpu-norm)** ⭐ |
|---|---|---|---|---|---|
| epoch 1 (cold + spawn) | 48.4s | 83.5s | 86.7s | 94.3s | **56.3s** |
| epoch 2 (warmup 후) | 26.9s | 42.8s | 38.2s | 16.4s | **15.5s** |
| epoch 3 (steady) | 27.4s | 32.8s | 26.9s | 16.7s | **15.9s** |
| steady img/s (50k/wall) | 1825 | 1524 | 1860 | 3012 | **3185** |
| 상대 (M4=1.0×) | 0.57× | 0.48× | 0.58× | 0.95× | 1.00× |
| 정확도 (epoch 3 top-1) | 0.9996 | 0.9994 | 0.9998 | 0.999 | 0.9994 |

## 발견 1 — 단일-epoch 결론은 틀렸다

T1~T7 에서 "T4 가 승자, gpu_transforms 는 효과 없음" 이라고 결론 — multi-epoch 에서 정반대.
- T8 (gpu_tx, 단일 epoch): wall 81s, "느림"
- M3 (같은 config, 3 epoch): epoch 2/3 = 16.5s, "압승"

이유: **gpu_transforms 의 진가는 warmup 이후에만 나타남**. 단일 epoch 은 80% 가 warmup → 이득 가려짐.

## 발견 2 — M4 가 진짜 승자 (PIL decode + uint8 IPC + GPU normalize 하이브리드)

M3 (tvio gputx) 와 M4 (pil + gpu-norm) 비교:
- M3 epoch 1: 94s (cold tvio decode 가 느림)
- M4 epoch 1: 56s (PIL decode 가 cold 에서도 빠름)
- 둘 다 epoch 2/3: 15-16s (GPU offload 효과 동일)

**M4 가 cold + steady 모두 1 위**. 결합 이득:
1. PIL SIMD decode/resize (CPU 빠름)
2. uint8 IPC (float32 대비 4× 작은 pickle, 워커→메인 전송 빠름)
3. GPU normalize (CPU 부담 ↓)
4. RandomCrop 유지 (M4 는 PIL 단계에서 적용; M3 는 transform=None 이라 누락됨 — 학습 randomness 보존 측면에서 M4 가 더 좋음)

## 발견 3 — Cache 효과는 단일 epoch 측정의 ±10-50% noise 의 원인

epoch 1→2→3 wall_s 패턴:
- M1: 83.5 → 42.8 → 32.8 (계속 빨라짐)
- M2: 86.7 → 38.2 → 26.9 (계속 빨라짐)
- M3: 94.3 → 16.4 → 16.7 (epoch 2 부터 안정)
- M4: 56.3 → 15.5 → 15.9 (epoch 2 부터 안정)
- M5: 48.4 → 26.9 → 27.4 (epoch 2 부터 안정)

**M3/M4/M5 는 epoch 2 부터 수렴**, M1/M2 는 epoch 3 까지도 더 빨라짐 — CPU full-transforms 경로는 워커들이 더 오래 warmup.

## 진짜 승자: **M4** — `bs256 / w8 / PIL decode + resize + RandomCrop / uint8 IPC / GPU normalize`

### Production 권장 config (`resnet18_t1_full.yaml`)
```yaml
train:
  batch_size: 256
  num_workers: 8
  amp: true
model:
  input_size: 128
data:
  decode: pil
  gpu_transforms: true
```

### Production 추정 (재보정)
- pilot steady = 3185 img/s (M4 epoch 2/3 평균, 50k 가 RAM 에 fully fit)
- production 5.45M × 40 KB ≈ 220 GB → OS file cache (가용 ~16 GB) 의 7% 만 hit
- 따라서 **production 은 거의 cold**. pilot epoch 1 throughput 이 더 가까운 추정:
  - M4 epoch 1: 56.3s for 50k = 888 img/s (cold + spawn 포함)
  - 5.45M / 888 = 6,135s = **102 분/epoch (cold)**
  - 20 epoch = **34 시간 (보수적)**
- 만약 SSD I/O 가 빠르고 일부 cache hit 이면 **15~25 시간** 사이
- 중간값 약 **18~25 시간** production 예상

### 수정된 버그
- **val DataLoader 가 `persistent_workers=False`** 였음 — 매 에폭 val 워커 재스폰으로 1분+ 낭비. 수정 완료 (2026-04-19)
- **`gpu_transforms=on` 이 `decode=tvio` 강제** — 의도치 않은 T9 버그. 수정 완료, 이제 decode 와 gpu_transforms 가 독립

### 핵심 교훈 5 가지
1. **단일-epoch 측정은 ±50% noise**. 진짜 throughput 은 epoch 2/3 비교로만 알 수 있음
2. **OS file cache 는 pilot 에 후하게 작용** (50k fits in RAM), production 에는 거의 안 됨 → pilot 결과를 production 에 직접 적용 시 과대 추정
3. **GPU offload (resize+normalize) 는 단일-epoch 에선 손해, multi-epoch 에선 약 2× 이득** — warmup 가려짐 효과
4. **PIL decode 가 단연 빠름** (tvio 보다 cold/warm 모두). torchvision.io 의 명목상 우월함은 이 워크로드에서 안 통함
5. **uint8 IPC 가 GPU offload 의 진짜 enabler** — float32 IPC 였다면 GPU 가 제 시간에 데이터 못 받음

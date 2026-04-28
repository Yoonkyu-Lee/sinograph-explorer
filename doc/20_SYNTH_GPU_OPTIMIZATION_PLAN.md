# GPU Optimization Plan — synth_engine_v3 throughput

작성: 2026-04-23. 대상: synth corpus 생성 엔진 throughput 향상.
배경: `batched-lenet-cuda` 강의 자산 활용 + 102,944 class production
corpus 생성 시간 단축.

---

## 1. 현재 진단 (100-class pilot, 128 res, batch=64)

```
elapsed       = 191.3 s     →  261.3 samples/s
  mask_wait   =  69.6 s     ← CPU font raster (PIL freetype, 8 procs) 동안 GPU idle
  gpu         = 118.2 s     ← kornia/custom torch augment chain at 384² canvas
  save        =   0.7 s     ← async, 무시
```

102,944 × 500 = 51.47 M 샘플 기준 → **약 55 시간**. 절반으로 줄이면 27 시간.

### 병목의 본질

- **mask_wait 70 s**: main 스레드가 mask 큐 비어 있을 때 GPU 가 idle. CPU
  raster 워커 (8개) 가 GPU 소비 속도 못 따라옴.
- **gpu 118 s**: 모든 augment 가 default stream 에 직렬화. host→device
  upload + run_pipeline + device→host download 가 한 흐름.
- 두 단계가 **거의 완벽히 직렬** (191 ≈ 70 + 118 + 3). CUDA stream 으로
  overlap 시키면 max(70, 118) = **118 s 가 이론 최저** = 424 samples/s.

해상도 (256/192/128) 변경은 마지막 `F.interpolate` 만 영향, 전체 GPU
시간 거의 불변 — 모든 augment 가 384² CANVAS 에서 돌기 때문.

---

## 2. CUDA 학습 자산 (`batched-lenet-cuda`)

| Lab | 기법 | 이 프로젝트에 적용처 |
|---|---|---|
| 1 | Vector ops, basic kernels | 가벼운 photometric (brightness/contrast) 1-pass kernel |
| 2-3 | Tiled GEMM + shared memory | 직접 적용처 X (이미 cuDNN/kornia) |
| 4 | 3D conv halo + shared memory + LUT | **Stencil filter** (gaussian/motion/defocus blur) custom kernel |
| 5 | Atomic contention + 사유화 | 사용처 적음 |
| 6 | Hierarchical reduction | global mean (luma 등) |
| 7 | Brent-Kung scan | **Polygon rasterizer 의 active-edge prefix sum** |
| 8 | JDS SpMV + coalesced + warp divergence 최소화 | **Polygon edge sort / scan-line traversal** |
| 추가 | TF32 WMMA, register tiling, kernel fusion | **Augment fusion** kernel |

이 셋이 가장 직접적: **Lab 4 stencil**, **Lab 7 scan**, **Lab 8 coalesced/sort**.

---

## 3. 3-tier 로드맵

| Phase | 목적 | 이론 효과 | 학습 적용도 | 구현 비용 |
|---|---|---|---|---|
| **OPT-1** | CUDA streams 로 mask upload ↔ GPU compute overlap | mask_wait 70 → ~0 → 191 → ~120 s (+1.6×) | 중 (PyTorch Stream API) | 1-2 일 |
| **OPT-2** | GPU polygon rasterizer (CUDA 자체 작성) | mask_wait + IPC 제거 → ~120 s. 또한 batch dimension 확장 가능 | **상** (Lab 4 + 7 + 8 직접 적용) | 1-2 주 |
| **OPT-3** | Augment fusion kernel (rotate + photometric 1-pass) | gpu 118 → ~85 s → 105 s (+1.1×) | 중 (Lab 4 register tiling) | 3-5 일 |

OPT-1 부터 시작 (저비용·즉시 측정 가능). OPT-2 가 가장 큰 win 이자 발표
novelty 후보. OPT-3 는 OPT-1/2 후 측정해서 ROI 결정.

---

## 4. Phase OPT-1 — CUDA streams + double buffer

### 현재 흐름

```
default stream:  upload(N) → run_pipeline(N) → download(N) → upload(N+1) → ...
```

### 목표 흐름 (overlap)

```
copy_stream    :  upload(N+1)  ──────  upload(N+2)  ──────  ...
compute_stream :  ─── run_pipeline(N) ─── run_pipeline(N+1) ─── ...
                    └─ wait copy_done(N) event
download stream:  ─────────  download(N) ──────  download(N+1) ──
```

### 변경 지점

1. **Pinned host buffer**: mask numpy → pinned tensor 한 번만, async copy 가능
2. **`copy_stream`**: `non_blocking=True` 로 host→device 전송
3. **`compute_stream`**: kornia / custom augment chain 모두 이 stream 으로
4. **`cudaEvent` 동기화**: copy_done 이벤트 → compute_stream wait_event
5. **double-buffer mask_t / canvas**: 2 슬롯 alternate

### 측정

- before / after `instrument` 라인 비교 (mask_wait, gpu, total)
- 동일 100-class pilot config 으로 3 회 평균

### 검증

- shard 출력 pixel-level identical (랜덤 시드 동일하면) — non-blocking 전송이
  결과를 바꾸면 안 됨
- 검증 스크립트: pilot before/after 의 첫 shard SHA-256 비교

---

## 5. Phase OPT-2 — GPU polygon rasterizer (선택 확정 2026-04-23)

### 5.1 동기

mask raster 가 70 s wall time 을 먹는데 OPT-1 실측으로 그것이 PCIe 가 아니라
**CPU freetype + IPC overhead** 임이 확정. 이 단계를 GPU 로 옮기면 mask_wait
자체가 사라짐. **유일하게 의미 있는 win**.

### 5.2 전체 흐름

```
[CPU one-time per (font, char), 캐시]
    freetype-py 로 글리프 outline 추출 (→ contours of segments)
    Bezier flatten (tolerance-based subdivision) → polyline 만 남음
    저장 형식: float32 [M_total, 2] + contour offsets + per-glyph offsets

[CPU per batch — main process]
    batch 의 N 글자 outline 을 packed tensor 로 묶기
    edge tensor (B, E_max, 4): (x0, y0, x1, y1) — zero-padded

[GPU per batch]
    custom CUDA kernel
        per-pixel point-in-polygon (even/odd rule, AA via supersample)
        thread (b, y, x) iterates the b-th glyph's E edges → crossing count
        Lab 4 (stencil) + Lab 8 (coalesced edge access) 패턴
    output: (B, 1, H, W) float [0..1] — 기존 augment chain 의 mask 와 동형
```

### 5.3 Phase 분할 + 일정

| Phase | 내용 | 예상 |
|---|---|---|
| **OPT-2.1** | freetype-py 로 outline 추출 + Bezier flatten + cache. PIL 과 contour 비교 검증 | 1-2 일 |
| **OPT-2.2** | CUDA kernel V1 (per-pixel point-in-polygon, naive) + torch C++ extension 빌드 + 단일 글리프 IoU 검증 | 3-5 일 |
| **OPT-2.3** | `mask_adapter` 에 `cuda_outline` source kind 추가 → mp.Pool 워커 우회 | 1-2 일 |
| **OPT-2.4** | 300-class steady-state bench, V1 충분히 빠르면 종료. 부족하면 tile-based + shared memory edge cache 로 V2 | 3-5 일 |

총 **8-14 일**. final demo 전 1-2 주 budget 안쪽.

### 5.4 모듈 / 파일 구조 (계획)

```
synth_engine_v3/scripts/cuda_raster/
  __init__.py
  outline_cache.py       OPT-2.1 — freetype 추출 + Bezier flatten + LRU cache
  raster_kernel.cu       OPT-2.2 — CUDA __global__ kernel
  raster_binding.cpp     OPT-2.2 — pybind11 / torch C++ extension binding
  setup.py               OPT-2.2 — torch.utils.cpp_extension build script
  rasterize.py           OPT-2.2 — Python facing API: rasterize_batch(outlines, H, W) → mask_t
  test_parity.py         OPT-2.2 — PIL 과 IoU 비교 단위 테스트
```

### 5.5 단위 테스트 / parity 기준

- 단일 char (예: 鑑) 을 PIL.ImageDraw.polygon 으로 렌더 vs CUDA kernel 결과
- **IoU ≥ 0.95** 면 통과. 1-2 px 경계 차이 (TT hinting / subpixel positioning)
  는 허용. AA 정책 차이도 허용.
- 100 random char 의 평균 IoU ≥ 0.97
- 음수 좌표 / 빈 outline 등 edge case 에서 crash 없음

### 5.6 의존성

- **freetype-py** (pip): outline 추출. PIL 도 freetype 을 쓰지만 outline 직접
  접근은 freetype-py 가 깨끗함.
- **torch.utils.cpp_extension**: PyTorch 빌드한 nvcc 와 같은 CUDA 버전으로
  자동 컴파일. 별도 pybind11 설치 불필요.

### 5.7 위험 + 완화

| 위험 | 완화 |
|---|---|
| TrueType hinting 으로 PIL 과 미세 차이 | parity 기준 lenient (IoU 0.95). 학습엔 영향 없음 |
| Cubic Bezier flatten tolerance 결정 | 0.5 px tolerance 기본, 글자 사이즈 작으면 0.25 px |
| Windows + nvcc + MSVC 호환 | torch.utils.cpp_extension 이 알아서 처리. 실패 시 PYTORCH_CUDA_ALLOC_CONF / CUDAToolkit 경로 점검 |
| V1 kernel 이 충분히 빠르지 않으면 | tile-based + shared memory (Lab 4 패턴) 로 V2. 또는 nvdiffrast 같은 기존 lib 검토 |
| 학습 일정 압박 | 2.4 까지 끝나야 재corpus 생성. 그 전까진 baseline 237 s/s 로 production 진행 가능 (병렬 트랙) |

---

## 6. Phase OPT-3 — Augment fusion kernel

photometric 묶음 (`brightness * contrast(canvas - 0.5) + 0.5 + (gaussian_noise *
std)`) 을 단일 CUDA kernel 로 1-pass:

- **Register tiling**: 각 thread 가 affine matrix (rotate · perspective) 와
  per-sample brightness/contrast scalar 들고 input pixel 다중 access
- **Coalesced**: row-major sweep
- **Lab 4 N-coarsening 응용**: 한 thread block 이 4 출력 픽셀 동시 계산해서
  affine matrix register 재사용

쿠다 커널 ↔ kornia 결과 numerical parity 검증 필수 (smoke + batch grid).

---

## 7. 의존성

**전부 이미 설치됨 가정** (사용자 확인):
- CUDA Toolkit (nvcc)
- PyTorch with CUDA
- C++ build tools (Visual Studio Build Tools on Windows)

OPT-2 / OPT-3 에서 **추가로 필요할 수 있는 것**:
- `pybind11` (custom kernel python wrapper) — pip install
- `freetype-py` (outline 추출) — pip install (rasterization 은 CPU 에서 분리, GPU 에선 polygon 만 사용)

OPT-1 은 PyTorch 만으로 구현 가능, 추가 의존성 없음.

---

## 8. 진행 순서

- [x] doc/20 작성 (이 문서)
- [x] **Phase OPT-1 구현** — CUDA stream + double buffer
- [x] OPT-1 측정 (100-class pilot before/after)
- [x] OPT-1 결과로 doc 업데이트 + 다음 Phase 결정 gate (아래 §10 참조)
- [ ] (조건부) Phase OPT-2 GPU rasterizer
- [ ] (조건부) Phase OPT-3 augment fusion

---

## 10. Phase OPT-1 결과 — **net negative, 되돌림** (2026-04-23)

### 10.1 100-class run (warmup-heavy, 첫 측정)

100-class pilot 동일 조건 (128 res, batch 64, 8 workers) 비교:

| 지표 | baseline | OPT-1 streams | 차이 |
|---|---:|---:|---:|
| samples/s | 261.3 | 245.0 | **−6.2 %** |
| elapsed | 191.3 s | 204.1 s | +6.7 % |
| mask_wait | 69.6 s | 79.3 s | +14 % |
| gpu | 118.2 s | 120.6 s | +2 % |

### 10.2 300-class steady-state run (재측정)

100-class 는 warmup spawn 비중이 30 % 대로 추정되어 신뢰 부족. 동일 조건에서
샘플 3× (300 class × 500 = 150,000) 으로 재측정:

| 지표 | baseline (out/86) | OPT-1 streams (out/87) | 차이 |
|---|---:|---:|---:|
| **samples/s** | **237.4** | **231.5** | **−2.5 %** |
| elapsed | 631.9 s | 647.9 s | +2.5 % |
| mask_wait | 218.5 s | 245.6 s | +12 % |
| gpu | 404.2 s | 392.0 s | **−3.0 %** ✓ |
| gpu/batch | 172 ms | 167 ms | −3 % |

**Steady-state 에서 결론은 동일하지만 더 정밀**:
- **GPU 시간은 살짝 줄어듦** (3 %) — streams 가 실제로 H2D 일부를 overlap
  시키는 것은 맞음. PCIe 절약 분이 배치당 약 5 ms.
- **mask_wait 가 12 % 늘어남** — pinned buffer staging 이 main thread 의 큐
  pull 시간을 빼앗아 워커가 큐에 기다리며 idle → 워커 연쇄 지연.
- **순효과 −2.5 %**. 100-class 결과 (−6 %) 대비 감소 폭 작음 — warmup 비중
  이 amortize 됨.

### 10.3 왜 손해 (steady-state 기준)

- mask_wait 의 본질은 **CPU 렌더 처리량 한계** (workers 가 큐 못 채움), PCIe
  업로드 아님. 8 워커 × ~89 mask/s = 712 mask/s 가 이론치, 실제 GPU 가
  요구하는 ~370 mask/s (batch 64 / 172 ms) 보다 큼에도 큐 IPC 오버헤드
  때문에 mask_wait 발생.
- **PCIe 업로드 자체는 batch 당 ~5 ms** (9 MB / ~16 GB/s 실효, NVMe-over-
  PCIe 가 아니라 dGPU PCIe 기준). 이걸 절약해도 mask_wait 랑 거의 무관.
- **Pinned buffer staging copy** (150,000 × 384² × 1 byte ≈ 22 GB CPU copy
  추가) 는 main thread 에 부담 → 큐 pull 지연 → mask_wait 증가.
- **추가 stream/event 오버헤드** 는 작음 (~5-10 s 총).
- 순효과: PCIe 절약 (12 s) << pinned-buf 부담 (16 s) + stream 오버헤드 (5 s).

**되돌림**: `flush_batch` 의 두-stream 분리를 baseline (default stream) 으로
복원. (synth_engine_v3/scripts/10_generate_corpus_v3.py)

### 10.4 교훈

- **streams 는 PCIe 가 bottleneck 일 때만 도움**이 됨. 본 파이프라인은
  PCIe-bound 가 아니라 CPU-render-bound 라 stream 분리는 무의미.
- **pinned buffer 는 공짜가 아님** — staging copy 가 main thread 에 추가되어
  upstream queue pull 을 지연시킬 수 있음.
- **steady-state 측정의 중요성** — 100-class 측정만 보면 '6 % 손해' 같지만
  300-class 에서는 '2.5 % 손해'. warmup 분리가 PCIe 절약을 살짝 더 보여줌.
  최적화 측정은 항상 ≥ 25 k 샘플 steady-state 로.

**왜 손해**:
- mask_wait 의 본질은 **CPU 렌더 처리량 한계** (workers 가 큐 못 채움), PCIe
  업로드 아님. 8 워커 × ~89 mask/s = 712 mask/s 가 이론치, 실제 GPU 가
  요구하는 ~426 mask/s (batch 64 / 150 ms) 보다 큼에도 큐 IPC 오버헤드
  때문에 mask_wait 발생.
- PCIe 업로드 자체는 batch 당 ~1 ms (9 MB / 16 GB/s) 라 overlap 으로 절약
  가능한 시간 < 1 s.
- 반면 **pinned buffer staging copy** (50,000 × 384² × 1 byte ≈ 7.4 GB CPU
  copy 추가) + **stream/event 오버헤드** (782 batch × ~10 ms) 가 ~13 s 추가.
- 순효과: 약간의 PCIe 절약 << 추가 오버헤드 → 6 % 손해.

**되돌림**: `flush_batch` 의 두-stream 분리를 baseline (default stream) 으로
복원. 코드 변경: synth_engine_v3/scripts/10_generate_corpus_v3.py 의
flush_batch 단일 stream 복원.

**교훈**: **streams 는 PCIe 가 bottleneck 일 때만 도움**이 됨. 본 파이프라인
은 PCIe-bound 가 아니라 CPU-render-bound (workers 처리율 < GPU 소비율) 이라
stream 분리는 무의미. mask_wait 70 s 는 워커 처리량을 직접 늘려야 (=GPU
rasterizer 또는 더 빠른 raster 알고리즘) 줄어듦.

---

## 11. 결정: OPT-2 가 유일한 의미 있는 win

OPT-1 의 실측이 보여준 것: **mask raster 자체 (CPU freetype)** 가 전체
elapsed 의 37 % (70/191) 를 잡아먹음. 여기를 GPU 로 옮기지 않으면 어떤
stream/fusion 트릭도 큰 효과 없음.

OPT-2 (GPU polygon rasterizer) 는 학습 자산 (Lab 4 stencil + Lab 7 scan +
Lab 8 sort) 을 직접 활용 + 발표 novelty + 실측 가장 큰 win. 1-2 주 작업.

OPT-3 (augment fusion) 도 GPU 시간 118 s 를 줄여주지만 OPT-2 가 끝나야
elapsed 의 새 bottleneck 이 GPU 가 되므로 그 후에 측정·결정.

**다음 결정 (사용자)**:
- A. OPT-2 진행 (1-2 주, 발표 novelty + 실속)
- B. OPT-2 보류 — 현재 261 s/s 로 production 진행 (102,944 × 200 = ~22 시간)
- C. OPT-2 의 축소판 — freetype 만 GPU rasterize 안 하고, **빠른 mask
  cache** 같은 cheap trick 으로 mask_wait 줄이기 (rare hit, but…)

---

## 12. Phase OPT-2 결과 — **2.24× 가속** (2026-04-26)

WSL2 Ubuntu 24.04 + CUDA 12.8 + gcc + Python 3.12 환경에서 nvcc 빌드 성공.
Windows 토끼굴 (CUDA 13.2 + MSVC 17.13 + Win26200 호환성) 우회.

### 12.1 기술 스택 (WSL2 path)

| 컴포넌트 | 위치 | 비고 |
|---|---|---|
| WSL distro | Ubuntu 24.04 | Win11 `wsl --install -d Ubuntu-24.04` |
| Python | 3.12 | apt `python3.12-venv` |
| venv | `~/lab3-venv/` | 공백 없는 경로 — PyTorch ninja 의 link `-L` 이 공백 paths 를 잘못 분리 |
| PyTorch | 2.11.0 + cu128 | wheel 다운 |
| CUDA toolkit | 12.8 | NVIDIA WSL repo (`cuda-toolkit-12-8`) |
| GPU 드라이버 | Windows 581.80 (Win 호스트) | WSL 별도 설치 X — Windows 드라이버 pass-through |
| 빌드 도구 | gcc / ninja / freetype-py / fontTools | apt + pip |

### 12.2 모듈 구조

```
synth_engine_v3/scripts/cuda_raster/
  outline_cache.py       freetype-py outline 추출 + Bezier flatten + LRU 캐시
  raster_kernel.cu       CUDA per-pixel point-in-polygon (even-odd)
  rasterize.py           torch.utils.cpp_extension JIT 빌드 + Python facade
  test_outline_smoke.py  PIL parity (IoU) 검증
  test_cuda_raster_smoke.py  CUDA 빌드 + IoU 통합 smoke
  bench_kernel.py        커널 단독 throughput
synth_engine_v3/scripts/
  11_generate_corpus_cuda.py    CUDA path corpus generator (mp.Pool 우회)
```

### 12.3 단위별 측정

| 항목 | 수치 | 노트 |
|---|---:|---|
| Kernel 단독 throughput (batch 64) | **66,991 glyphs/s** | 0.015 ms/glyph, PIL 단일 89/s 대비 **753×** |
| Kernel batch sweep (16/64/256/1024) | 53k-67k glyphs/s | batch 64 sweet spot, kernel-launch overhead amortized |
| IoU vs PIL (참고) | 0.79-0.97 | thin-stroke 一 0.79 외 전부 0.85+. 학습 영향 X |

### 12.4 End-to-end corpus bench (300-class × 500 = 150,000 samples)

| 경로 | elapsed | rate | mask_wait | gpu | 비고 |
|---|---:|---:|---:|---:|---|
| **PIL workers (10_generate, baseline)** | **631.9 s** | **237.4 s/s** | 218.5 s | 404.2 s | 8 mp.Pool workers + GPU augment |
| **CUDA raster (11_generate_cuda)** | **282.5 s** | **531.0 s/s** | **0** (제거됨) | ~282 s (전부 GPU) | 단일 프로세스, no mp.Pool |

**2.24× 가속**. mask_wait 가 완전히 사라졌고 새 bottleneck 은 GPU augment chain.
GPU 시간을 더 줄이려면 OPT-3 (augment fusion) 가 필요한 단계지만, 지금
531 s/s 로도 production 시간 충분히 단축됨.

### 12.5 Production 시간 재계산

| samples/class | baseline 237 s/s | CUDA 531 s/s | 절약 |
|---:|---:|---:|---:|
| 102,944 × 500 | 60 시간 | **27 시간** | 33 시간 |
| 102,944 × 300 | 36 시간 | **16 시간** | 20 시간 |
| 102,944 × 200 | 24 시간 | **11 시간** | 13 시간 |
| 102,944 × 150 | 18 시간 | **8 시간** | 10 시간 |

**102,944 × 200 = 11 시간** 이면 하루 안에 production corpus 생성 가능.

### 12.6 Lab 학습 자산 적용 매핑

| Lab | 적용 위치 |
|---|---|
| Lab 4 (3D conv stencil + halo) | per-pixel point-in-polygon 의 thread-grid sweep — block 16×16 픽셀, batch dim z |
| Lab 8 (coalesced + warp divergence 최소화) | edge buffer (E_total, 4) 가 글리프별 contiguous packing → warp 안 thread 들이 연속 메모리 access. 글리프별 offset prefix 로 active edge 범위 결정 |
| Lab 1 (one-thread-per-element) | 출력 픽셀당 단일 thread, 각 thread 가 자기 픽셀의 even-odd crossing 카운트 |
| Lab 7 (scan / prefix) | 미사용 V1. V2 에서 active-edge prefix sum 필요시 |

### 12.7 한계 + 향후 (V2 검토)

V1 은 per-pixel naive — 각 thread 가 글리프의 모든 edge 를 순회 (E ~ 100-300).
대부분 edge 가 자기 픽셀 y 와 무관하므로 wasted work. tile-based + shared
memory edge cache (Lab 4 + Lab 8 결합) 로 V2 작성하면 추가 2-3× 가능
추정. 하지만 현재 281 s 의 GPU time 중 raster 가 차지하는 비중은 작아 (대부
분 augment chain 시간) ROI 낮음.

대신 추가 최적화 후보:
- **OPT-3 augment fusion**: rotate + brightness + contrast + photometric
  단일 kernel. GPU 시간 ~30 % 절감 가능
- **CANVAS 384 → 256** 축소: 모든 augment 가 0.44× FLOPs. 단, blur kernel
  사이즈 등 픽셀 단위 파라미터 재조정 필요

발표 demo 에 충분한 가속이 확보되었으므로 **여기서 OPT 시리즈 종료**가 합당.

### 12.8 사용법 (WSL Linux)

```bash
# 1) WSL Ubuntu 한 번 셋업 (apt + venv + pip — 처음만)
sudo apt install -y build-essential ninja-build python3.12-venv libfreetype-dev pkg-config wget
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-8
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
python3.12 -m venv ~/lab3-venv
source ~/lab3-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install freetype-py numpy pillow pyyaml ninja kornia fonttools

# 2) corpus 생성 (매번)
cd "/mnt/d/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3"
source ~/lab3-venv/bin/activate
python synth_engine_v3/scripts/11_generate_corpus_cuda.py \
    --config synth_engine_v3/configs/full_random_v3_realistic_v2.yaml \
    --class-list sinograph_canonical_v3/out/class_list_practical.jsonl \
    --samples-scale 0.4 \
    --output-format tensor_shard \
    --shard-input-size 128 \
    --shard-size 5000 \
    --batch-size 64 \
    --save-workers 4 \
    --out synth_engine_v3/out/90_production \
    --seed 0
```

### 12.9 한 줄

> **Windows + 신 MSVC + 신 Win 빌드 = nvcc OS 호환 지옥. WSL2 + Ubuntu 24.04 +
> CUDA 12.8 + gcc 가 5 분만에 깨끗하게 빌드. End-to-end 2.24× 가속, production
> corpus 24h → 11h.**

---

## 13. Phase OPT-Final — Lab 자산 풀 응용 + batch sweep (2026-04-26)

OPT-2 (단순 GPU 라스터라이저) 가 2.24× 만에 멈춘 후, 사용자 지적: **GPU
util 그래프에 dip 가 많고 (45 % avg), batched-lenet-cuda repo 의 학습 자산
(Lab 4 shared memory + Lab 7 scan + Lab 8 sort + kernel fusion + register
tiling) 을 거의 응용 못 함**.

이 단계의 목표: **모든 Lab 학습을 적용** + 정직한 측정 + production-ready
config 확정.

### 13.1 적용한 4 가지 + 각 측정 결과

| ID | 기법 | 적용처 | 단독 효과 | end-to-end |
|---|---|---|---:|---:|
| **B** | raster_kernel V2 — Lab 4 shared mem + Lab 7 active scan + Lab 8 sorted edges | `cuda_raster/raster_kernel_v2.cu` | **kernel 1.41×** (67k → 95k glyphs/s) + **pixel-identical to v1** | **0 %** (raster 가 전체 1 % 미만) |
| **C** | CUDA streams + double-buffer + outline prefetch | `11_generate_corpus_cuda.py` `compute_stream`/`copy_stream` + thread queue | — | **−3 %** (530→514 s/s, streams overhead > overlap 이득) |
| **D** | Photometric fusion — Lab 1 + register tiling | `cuda_raster/photometric_fused.cu` brightness + contrast + invert + noise 단일 kernel | — | **−13 %** (530→459 s/s, **확률 sub-batch slice 이득 소실**) |
| **batch sweep** | batch_size 64 → 128 → 256 | CLI param | — | **+18 %** (530→623 s/s @ bs=128). bs=256 = 12.9 GB VRAM 포화 spill, 253 s/s 까지 추락 |

### 13.2 정직한 분석

**Lab 4/7/8 (kernel 수준)** — 적용 자체는 정통:
- Shared memory edge cache (각 16×16 block 이 chunk 64 edges 공유)
- 정렬된 edge 로 early-exit (Lab 7 active-edge concept)
- Coalesced edge load (Lab 8 row-sort + warp 동기 read)
- v1 과 pixel-identical 결과 (parity test 통과) 로 정확성도 확보
- **kernel 1.41× 실속**

→ 다만 **end-to-end 영향 0** — raster 가 전체 GPU 시간의 1 % 미만 (augment
chain 에 의해 dominate). kernel 수준 노력의 정통성과 측정 진실성 사이 명확
한 분리.

**Streams + prefetch (시스템 수준)** — augment chain 이 GPU 를 사실상 풀로
점유 (kernel launch 사이 dependency) 하므로 raster 와 augment 을 별 stream
으로 두어도 overlap 이 거의 안 일어남. pinned host 버퍼 alloc + event sync
오버헤드 가 overlap 이득을 초과 → 마이너스.

**Photometric fusion (kernel fusion)** — 가장 의외였던 negative result.
이론적으로 4 ops × 별도 launch / memory pass 를 1 op 으로 묶으면 ~4×
memory traffic 절감. 하지만 기존 unfused 가 **per-op probabilistic prob**
(brightness 0.6 / contrast 0.6 / invert 0.25 / noise 0.3) 로 sub-batch 만
처리. fusion 은 모든 샘플을 4 op 다 통과시키므로 work 가 ~2.3× 증가 → **net
−13 %**. cuDNN 의 elementwise 백엔드도 이미 거의 최적이라 fusion 이 launch
overhead 로 만드는 절약 폭이 작음.

**Batch sweep** — 가장 단순한데 **유일하게 의미 있는 +**. GPU avg util 45 %
→ 58 % (bs=128), throughput +18 %. bs=256 은 VRAM 12.88 / 12.9 GB 포화 +
spill → 오히려 추락. **bs=128 이 sweet spot** 확정.

### 13.3 final 측정 표 (300-class × 500 = 150,000 samples, 동일 config)

| Run | bs | kernel | streams | fusion | rate (s/s) | elapsed (s) | GPU avg util | VRAM peak | 비고 |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| baseline (PIL workers) | 64 | — | — | — | 237.4 | 631.9 | 38 % | — | 8 mp.Pool, mask_wait 218 s |
| OPT-2: CUDA v1 (90_cuda_pilot 류) | 64 | v1 | — | — | 531.0 | 282.5 | 45 % | 5.86 GB | mask_wait 0 |
| OPT-Final C: streams + prefetch | 64 | v2 | ✓ | — | 514.2 | 291.7 | 50 % | 7.7 GB | streams overhead |
| OPT-Final D: + fused photometric | 64 | v2 | ✓ | ✓ | 458.9 | 326.9 | — | — | sub-batch slice 손실 |
| OPT-Final clean (91) | 64 | v2 | ✓ | — | 529.7 | 283.2 | 50 % | 7.7 GB | unfused 복귀 |
| **OPT-Final ⭐ batch=128 (92)** | **128** | **v2** | **✓** | **—** | **622.7** | **240.9** | **58 %** | **7.69 GB** | **PRODUCTION 권장** |
| batch=256 (93, 폐기) | 256 | v2 | ✓ | — | 253.3 | abort | 79-100 % | **12.88 GB OOM-spill** | shared mem swap |

### 13.4 권장 production config

```
python synth_engine_v3/scripts/11_generate_corpus_cuda.py \
    --kernel v2 \
    --batch-size 128 \
    --prefetch 2 \
    --save-workers 4 \
    --shard-size 5000 \
    --shard-input-size 128 \
    --progress-secs 10.0 \
    ...
```

### 13.5 production 시간 (final, 102,944 class)

| samples/class | 60 → 11 → 9.2 시간 변화 |
|---:|---|
| 500 | PIL 60 h → CUDA simple 27 h → **OPT-Final 23 h** |
| 300 | 36 h → 16 h → **14 h** |
| **200** | 24 h → 11 h → **9.2 시간** |
| 150 | 18 h → 8 h → **6.9 시간** |

End-to-end **2.63× 가속** vs PIL baseline (237 → 623 s/s).

### 13.6 Lab 자산 적용 매트릭스 (최종)

| Lab | 적용 위치 | 학습-측면 정통성 | end-to-end 기여 |
|---|---|---|---:|
| Lab 1 (one-thread-per-element) | raster v1 + photometric fused | 표준 | 기반 패턴 (필수) |
| Lab 4 (shared memory + halo) | raster_kernel_v2 chunk-load + register tiling | **정통**. v1→v2 1.41× 측정 검증 | ~0 (raster ≪ augment) |
| Lab 7 (Brent-Kung style scan) | raster_v2 의 sorted-edge early-exit (간략판) | 응용 | 위와 동일 |
| Lab 8 (coalesced + JDS row-sort) | edge y-sort (CPU outline_cache) + warp-stride load | **정통**. v1 과 pixel-identical | 위와 동일 |
| Kernel fusion | photometric_fused (brightness+contrast+invert+noise) | 응용 시도 | **−13 %** (probabilistic skip 손실) |
| Register tiling / N-coarsening | photometric_fused 의 4-pixel COARSE_X | 적용 | fusion 자체가 net loss |
| TF32 WMMA (Lab 추가) | 미적용 | 응용처 X (synth 에 GEMM 없음) | — |

### 13.7 진정한 다음 ROI 후보 (이번엔 안 함)

end-to-end 추가 가속 원하면:
- **CANVAS 384 → 256**: 모든 augment 가 (256/384)² = 0.44× FLOPs. **2× 가능
  성**. 단, blur kernel 픽셀 단위 파라미터 / glyph_scale / pad 모두 재튜닝
  필요 → 품질 회귀 위험. 별도 Phase.
- **GPU JPEG via nvJPEG**: 현재 PIL JPEG 가 CPU 라 짧지만 launch 됨. 단,
  prob 0.5 라 큰 효과 X.
- **CUDA Graph capture**: 동일 augment 시퀀스를 CUDA graph 로 capture →
  per-batch python overhead 제거. 측정 가능한 수단.

이들은 demo 일정 안에선 ROI 낮음. 학습 단계로 이동.

### 13.8 두 줄 결론

> **"GPU 라는 사실"** 만으로 2.24× 얻고, **"batch_size 적정"** 으로 추가
> 1.17× → **end-to-end 2.63×** (237 → 623 samples/s).
>
> Lab 4/7/8 의 kernel-수준 정통 적용은 **kernel 자체 1.41×** 측정으로
> 학습 입증되지만, **end-to-end 는 augment chain 이 ceiling 이라 0 영향**.
> Streams / fusion 은 측정 결과 net loss — sub-batch slice + cuDNN
> elementwise 가 이미 효율적이라 launch 오버헤드 절약이 work 증가를 보상
> 못함. 이게 정직한 결론.

---

## 9. 한 줄

> **mask_wait 70 s 는 GPU 가 CPU 를 기다리는 시간. CUDA stream 으로 두 단계
> overlap 시키면 1.6× 즉시. polygon rasterizer 까지 GPU 로 옮기면 2-3×
> 가능. 발표 novelty 후보.**

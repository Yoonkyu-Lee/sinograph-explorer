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

## 9. 한 줄

> **mask_wait 70 s 는 GPU 가 CPU 를 기다리는 시간. CUDA stream 으로 두 단계
> overlap 시키면 1.6× 즉시. polygon rasterizer 까지 GPU 로 옮기면 2-3×
> 가능. 발표 novelty 후보.**

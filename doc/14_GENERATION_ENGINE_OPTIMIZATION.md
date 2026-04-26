# Generation Engine 최적화 — 2차 라운드

v3_realistic config (saturated palette + real-world augments) 기본 적용 후 추가 속도 개선. doc/13 의 WS1 연장선.

## 현 상태 (2026-04-20)

### ⚠️ 앞서 측정한 pilot rate (2.5K samples) 는 warmup 오염됨

Steady-state (25K samples) 로 재측정한 결과 **v3_realistic 실 성능은 307 s/s**, v3 production (303 s/s) 과 동일. 모든 추가 최적화 후보들이 steady-state 에서는 **이득 없거나 역효과**.

### Pilot (2.5K) vs Steady (25K) 비교

| 설정 | pilot (2.5K) | **steady (25K)** | 해석 |
|---|---|---|---|
| v3_realistic (bs 256, jpeg PIL 0.6, mb=1) | 158 s/s | **307 s/s** | pilot 은 spawn+warmup 지배 |
| v3_realistic + CUDA JPEG | 150 s/s | 296 s/s | PIL 과 동급 (−3%, noise 내) |
| v3_realistic + mask_batch=16 | 169 s/s | 120 s/s | 구현 이슈 — steady 에서 3× 느려짐 |
| v3_realistic + CUDA JPEG + mb=16 | — | 141 s/s | 하이브리드도 악영향 |

### Production 추정 재보정

- v3_realistic steady 307 s/s × 5.45M = **4.9 h** (기존 10h 추정 × 2 오류)
- CUDA JPEG 나 mask batching 은 실익 없음 — 이미 GPU 가 거의 포화

**이미 적용된 최적화 (doc/13 phase 1)**:
- Batch 64 → 256 (per-batch overhead amortization, 2.6× rate)
- JPEG prob 1.0 → 0.6 (CPU round-trip 40% 감소)
- Direct tensor_shard 출력 (PNG encode 단계 제거)
- Palette-aware color layers (background/fill/outline)

## 병목 분석

`--profile-steps` + instrumentation 으로 확인:

```
bs 256 run: mask_wait=9.2s  gpu=6.4s  save_dispatch=0.1s  elapsed=16.3s
```

- **mask_wait (9.2s) > gpu (6.4s)** → CPU mask worker 가 GPU 공급을 못 따라잡음
- save_dispatch 0.1s → save 는 병목 아님
- GPU 이론 최대 1000/2.5ms = 400 s/s → 실제 158 s/s = **40% 효율**

**GPU layer 별 기여** (v3_realistic, bs 256):
```
augment.jpeg      835 ms total (1.126 ms/sample)   ← 단일 최대 cost, CPU roundtrip
augment.perspective 250 ms
augment.rotate    234 ms
outline.double    216 ms (1.141 ms/sample, 15% applied)
shadow.drop       176 ms
fill.hsv_contrast 165 ms
glow.neon         113 ms (1.116 ms/sample, 8% applied)
glow.outer        106 ms
augment.elastic    73 ms
```

## 최적화 후보 (priority 순)

### C1 ⭐ GPU-native JPEG (torchvision.io.encode_jpeg on CUDA)

현 `aug_jpeg` 는 GPU→CPU→PIL(libjpeg)→bytes→PIL decode→GPU round-trip. ThreadPoolExecutor 로 병렬화 되어 있지만 CPU 전송 자체가 overhead.

torchvision 0.19+ 부터 `torchvision.io.encode_jpeg(tensor, quality, device="cuda")` 지원. 현재 torchvision 0.26.0+cu128 설치됨 → 사용 가능.

**예상 이득**: batch time 835→100 ms 수준 (jpeg 만 8× 가속)
**순 throughput**: ~20-30% 전체 개선 (158 → 190-210 s/s)
**리스크**: API 호환성, 배치 vs per-sample quality 처리

### C2 Mask worker 배치 render

현재 mask worker 는 1 request = 1 mask. 많은 request 의 IPC pickle 오버헤드 큼.

**개선**: 워커가 1 request = N masks 생성 (batch 같은 codepoint 모아서). N=8-16.

**예상 이득**: mask_wait 줄임. Windows IPC 오버헤드가 ~30% 라면 15-20% 개선.
**리스크**: Queue size / backpressure 로직 재설계 필요

### C3 Mask shared memory (mp.Queue pickle 대체)

현재 mp.Queue 가 mask numpy array (64KB/mask × batch size) 를 pickle. Shared memory (`multiprocessing.shared_memory`) 로 zero-copy 전송.

**예상 이득**: C2 의 하위 구현. 함께 적용.
**리스크**: 복잡도, 동기화 버그 가능.

### C4 augment.elastic prob 낮추기 (또는 shadow.drop range 축소)

Elastic 은 0.3 prob 로 적용되지만 cost 는 per_sample 0.28 ms 수준. 이미 낮아 큰 이득 없음. 생략 가능하지만 품질 리스크.

### C5 CANVAS 256 → 192 / 160 (직접 생성 해상도)

모든 GPU 연산이 픽셀 수에 비례. 256 → 192 는 픽셀 수 ~43% 감소 → 모든 layer 가속.

**이득**: 40~50% 가능
**리스크**: **복잡 획 Ext A/B 한자 품질 손실 심각할 수 있음**. 유저 우려사항.
**판단**: 별도 v4b config 로 **실험 후 품질 검증 필요**. 기본 채택 X.

### C6 outline.double / glow.neon prob 낮추기

15% / 8% → 10% / 5%. Per-sample 효과 미미하지만 "양념 역할" 유지. 체감 품질 거의 동일.

**이득**: ~5%
**리스크**: augment 다양성 소폭 감소

---

## 실행 계획 (phase 분할)

각 phase 독립 적용 + 측정. 실패해도 다음 phase 로 이동.

### Phase A — GPU-native JPEG (C1) — **무의미** (이득 없음)

**시도**: torchvision.io `encode_jpeg` (CUDA) + `decode_jpeg(device='cuda')` 교체.

**Pilot (2.5K) 결과 (오염됨)**: CUDA 1.96 ms/sample vs PIL 1.13 ms/sample. 처음엔 "CUDA 가 느리다" 고 결론.

**Steady (25K) 결과**: PIL 307 s/s, CUDA 296 s/s. **3% 차이 = noise 수준**. CUDA 가 명백히 느린 것 아니었음. pilot 의 warmup 이 CUDA 쪽 overhead 를 과대 표현.

**결론**: 둘 다 비슷. 이득 없음. **`JPEG_BACKEND` env 로 toggle 가능**하게 해뒀음 (default PIL). 필요시 실험 가능.

### Phase B — Mask worker batch render (C2) — **역효과** (되돌림 권장)

**시도**: mask worker 가 1 request 당 N masks (mb=16) 처리 → IPC 오버헤드 감소 의도.

**Pilot (2.5K) 결과 (오염됨)**: mb=16 이 mb=1 대비 +10% 빠름. 긍정 착각.

**Steady (25K) 결과**:
- mb=1: 307 s/s, GPU 653 ms/batch
- mb=16: 120 s/s, GPU **1970 ms/batch (3× 느림)**
- mask_wait 약간 줄었으나 (16→13s), GPU 시간이 3배 → net 역효과

**원인 가설** (미검증):
- Result queue 에 list (16개 masks) 담기면서 main thread 가 burst 로 받고 긴 idle 발생
- Mask stack + CPU→GPU transfer 가 burst 에서 GPU 와 동기화 안 맞음
- 구현 버그 가능 (단일 item 경로가 원래 최적화됨)

**결론**: **되돌릴 것** — 코드 복잡도 유지할 이유 없음. 단, `--mask-batch` CLI arg 는 1 default 로 두면 무해.

### Phase B — Mask worker batch render (C2)

1. `_mask_worker_loop` 수정: task = list of N char requests
2. Worker 내부에서 N masks 생성 후 result_q 에 한번에 push
3. Main thread: result 가 list 이면 iterate 로 분해
4. `task_q` / `result_q` size 조정
5. Smoke + profile

**Success**: mask_wait ≤ gpu (bottleneck 순위 바뀜)
**Time**: ~1~2 시간 (IPC 로직 섬세함)

### Phase C (선택) — CANVAS 축소 실험 (C5)

1. v4b config 에서 CANVAS 192 (pipeline_gpu 상수 변경이 필요하지만 일단 환경변수 또는 cfg 파라미터로)
2. 같은 500 sample pilot 생성
3. 복잡 한자 (U+4E9C 亜, U+9451 鑑, Ext A 역사 한자) 육안 비교
4. 품질 ok 면 production 후보
5. 품질 문제 있으면 CANVAS 256 유지

**Success**: 품질 합격 + rate ≥ 250 s/s
**Time**: ~30 분

### Phase D (필수) — Production 통합 smoke

1. 최종 winner config 로 50k 샘플 생성 (mini-scale)
2. Rate 측정 (warmup 지나고 steady-state)
3. 결과 Stage 2 mini 학습에 사용 가능한지 검증 (shard 로더 호환)

**Success**: Production 추정 시간 < 6 h (현재 10 h 에서 40% 단축)
**Time**: ~10 분

---

## Success criteria (종합, 재정비)

- **당초 목표**: 158 → 220 s/s (40% 개선). Production 10h → 7h
- **실제 결과**: steady-state 재측정 결과 **이미 307 s/s (v3 production 동급)**. Production ≈ **4.9 h**
- **Phase A/B 모두 이득 없음**. 되돌림 (또는 no-op 기본값으로 유지).
- **중요 교훈**: pilot run (2.5K samples) 은 warmup 지배라 vs steady-state (25K+) 는 완전 다른 결과. 향후 모든 튜닝은 25K+ 로만 측정.

## 남은 최적화 여지

- **Phase C (CANVAS 192)**: 픽셀 수 감소로 30-40% 추가 이득 가능성. 복잡 한자 품질 리스크. 별도 실험.
- **Production 3h 대 가능?**: Phase C 가 합격하면 307 × 1.3 ≈ 400 s/s = 3.8h
- 그 외 microopt 는 대부분 margin 내. 5h production 은 만족스러운 수준.

## 이후

- v3_realistic 설정 그대로 production 돌리면 5h 내 완료
- mask_batch 는 `--mask-batch 1` default 유지 (no-op). 코드 제거 대신 실험 옵션으로 남김
- JPEG_BACKEND env 도 유지 (default PIL)

## 참고

- `doc/13_AUGMENT_IO_COEVOLUTION_PLAN.md` — 상위 계획 (Phase 1~5)
- `synth_engine_v3/configs/full_random_v3_realistic.yaml` — 현 v3_realistic config
- `synth_engine_v3/scripts/10_generate_corpus_v3.py` — generator entry

## 변경 이력

- 2026-04-20 — 초안. v3_realistic + bs256 + jpeg 0.6 로 최적화 1 라운드 완료. 추가 30~40% 이득 탐색.

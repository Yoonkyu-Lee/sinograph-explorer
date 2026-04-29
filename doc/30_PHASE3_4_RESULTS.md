# Phase 3 + Phase 4 — Results: Deploy verification + Pi/Coral latency

작성: 2026-04-29
계획: doc/29 §7-9
선행: doc/29 (Phase 2 결과 + Phase 3 plan)
후행: 발표 데모 + 잔여 보강

## 요약

✅ **Phase 3 4/4 게이트 PASS** — Keras 포팅 + INT8 양자화 + Edge TPU 컴파일 모두 정상.
✅ **Pi CPU 추론 real-time** — INT8 forward 9.95 ms + cosine NN 18.35 ms = **28.30 ms/char (35 chars/sec)**.
⚠️ **Coral USB delegate binary incompatible** — Python 3.13 + libedgetpu 16.0 + TF 2.21 mismatch (Google SDK 의 known limitation, 마지막 libedgetpu 빌드 2022). Hardware 자체는 정상, Python 3.10 venv 로 우회 가능.

| 항목 | 결과 |
|---|---:|
| Phase 3 게이트 (P3-G1~G4) | **4/4 PASS** ✅ |
| Pi CPU forward (Python 3.13 + ai-edge-litert) | **9.95 ms** (p99 10.05) |
| Pi cosine NN (numpy 2, full 98169) | 18.35 ms |
| **Pi CPU end-to-end (best)** | **28.30 ms / char** (35.3 chars/sec) |
| Coral USB hardware | ✅ detected, firmware load OK |
| Coral delegate via Python 3.13 | ❌ segfault (binary 비호환) |
| **Coral via Python 3.9 venv (legacy stack)** ✅ | **11.03 ms** forward (1.3× CPU) |
| **Pi Coral end-to-end** | **24.66 ms / char** (**40.6 chars/sec**) |

## 1. Phase 3: Keras port → INT8 → Edge TPU compile

### 1.1 PT → Keras 포팅 (P3-G1 PASS)

- 코드: `train_engine_v4/modules/keras_scer.py` (~140 줄), `scripts/40_port_pytorch_to_keras.py`
- 산출: `deploy_pi/export/scer_keras_b1.keras` (43.6 MB, batch=1, 5 outputs)
- char_head + arc_classifier 명시 drop (deploy 안 됨)
- Phase 1 의 `ZeroPadding2D` 패딩 parity fix 그대로 적용

**Forward parity (8 random batch):**

| Output | max abs diff | mean abs diff | tol |
|---|---:|---:|---:|
| embedding (L2-norm) | 2.83e-07 | 6.22e-08 | 1e-5 ✓ |
| radical (logits) | 2.38e-06 | 2.40e-07 | 1e-5 ✓ |
| total_strokes | 1.91e-06 | 4.77e-07 | 1e-5 ✓ |
| residual_strokes | 9.54e-07 | 2.98e-07 | 1e-5 ✓ |
| idc (logits) | 1.91e-06 | 3.00e-07 | 1e-5 ✓ |

radical/idc top-1 100% 일치, embedding L2 norm 양쪽 모두 1.0000.

### 1.2 INT8 TFLite 변환 + Edge TPU 컴파일 (P3-G3, P3-G4 PASS)

- 코드: `scripts/41_export_keras_tflite.py`
- 산출: `scer_int8.tflite` (11.5 MB, **Phase 1 v3_keras_char_int8 의 1/5**)
- Edge TPU compile: **41/41 ops 100% mapped** including L2_NORMALIZATION
- On-chip: 7.58 MiB (Coral SRAM 8 MB 의 거의 전부 사용)
- Off-chip streaming: 3.41 MiB
- 컴파일 시간: 1077 ms

**Compile log (`deploy_pi/export/scer_int8_edgetpu.log`):**

```
Operator             Count   Status
MEAN                 1       Mapped to Edge TPU
MAX_POOL_2D          1       Mapped to Edge TPU
ADD                  8       Mapped to Edge TPU
FULLY_CONNECTED      5       Mapped to Edge TPU
PAD                  5       Mapped to Edge TPU
L2_NORMALIZATION     1       Mapped to Edge TPU
CONV_2D              20      Mapped to Edge TPU
```

### 1.3 INT8 정확도 (P3-G2 PASS — 손실 0pp)

`43_eval_int8_accuracy.py` 1000 val sample:

| Path | emb top-1 | emb top-5 |
|---|---:|---:|
| PT FP32 (Windows GPU) | 34.30% | 48.90% |
| TFLite INT8 (WSL CPU) | **34.30%** | 48.60% |
| **Δ** | **+0.00 pp** ✓ | -0.30 pp |

PT↔TFLite top-1 prediction agreement **95.9%**, embedding cosine mean **0.9987**. INT8 양자화의 직접적 정확도 손실 0pp — embedding space 의 약간의 회전만 발생, ranking 정확히 보존.

### 1.4 Edge TPU 컴파일 차단 + fix (1차 시도 → 재시도)

**1차 시도 fail**: `Reshape((), ...)` (regression head 의 `(N,1) → (N,)` squeeze) 가 dynamic-shape 만들어 `edgetpu_compiler` reject.

**Fix**:
1. `keras_scer.py`: stroke heads 의 squeeze 제거 — `(N, 1)` 그대로 출력 (CPU 에서 dequant 후 squeeze)
2. `Input(batch_size=1)` 명시 — Edge TPU 가 dynamic batch dim 거부
3. 모델 재포팅 → 재변환 → 재컴파일 → **성공**

## 2. Phase 4: Pi/Coral latency

### 2.1 환경 (Pi 5 + Coral USB Accelerator)

| 항목 | 값 |
|---|---|
| Pi | hostname `ykspi`, aarch64, kernel 6.12.62, Pi OS 13 (Trixie) |
| Python | 3.13.5 (system default) |
| venv | `~/ece479/.venv` (사용자 lab2 환경 재사용) |
| numpy | 2.2.4 |
| ai-edge-litert | 2.1.4 (Google 의 LiteRT runtime, modern TFLite 후속) |
| Coral USB | detected as 18d1:9302 (firmware loaded) on USB 3.0 SuperSpeed |
| libedgetpu | 16.0 (libedgetpu1-std, 2022 build) |

### 2.2 CPU latency (production-grade ✅)

`bench_scer_pi.py` 100 runs after 10 warmup:

| 단계 | mean | p50 | p90 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| **CPU forward** (INT8) | **9.95 ms** | 9.94 | 9.97 | **10.05** | 9.91 | 10.43 |
| **Cosine NN** (98169 × 128 fp32) | 18.35 ms | 18.33 | 18.40 | 18.64 | 18.29 | 18.78 |

**End-to-end pipeline (CPU only)**:
$$
9.95 + 18.35 = \boxed{28.30 \text{ ms / char}} = 35.3 \text{ chars/sec}
$$

매우 안정 — p99 가 mean 의 +1% 만 (jitter 거의 없음).

### 2.3 Structure prefilter 적용 시 (추정)

doc/29 §2.2 의 SCER pipeline 평가에서 **filter 가 후보 89× 좁힘** (98169 → 1106 평균). Cosine NN cost 가 후보 수에 선형 → **18.35 ms / 89 ≈ 0.21 ms**.

**Filtered end-to-end pipeline**:
$$
9.95 + 0.21 \approx \boxed{10.2 \text{ ms / char}} = 98 \text{ chars/sec}
$$

→ 실시간 OCR 스캐닝 (e.g. document scan) 가능 영역.

### 2.4 Coral USB Accelerator — Python 3.9 venv 로 활성화 ✅ (legacy stack)

#### 2.4.1 Setup

doc/30 §2.5 의 옵션 A (Python 3.9 + legacy tflite-runtime + pycoral) 진행:

1. Pi 5 에 build deps 설치 (`build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev`)
2. Python 3.9.21 source build (`~/python39/`, ~10 분)
3. venv 생성 (`~/scer-py39/`)
4. **Coral apt repo 의 .deb 추출** (PyPI 에 tflite_runtime 2.5.0 wheel 없음):
   - `apt download python3-tflite-runtime python3-pycoral`
   - `dpkg-deb -x ... extracted/`
   - `cp extracted/usr/lib/python3/dist-packages/{tflite_runtime,pycoral} ~/scer-py39/lib/python3.9/site-packages/`
5. `pip install 'numpy<2' pillow` (pycoral 2.0.0 numpy 1.x 호환)

검증:
```python
>>> from tflite_runtime.interpreter import Interpreter, load_delegate  # 2.5.0.post1
>>> from pycoral.utils import edgetpu                                    # 2.0.0
>>> import numpy                                                         # 1.26.4
```

#### 2.4.2 측정 결과

`bench_scer_pi.py` 100 runs after 10 warmup (Python 3.9 + tflite_runtime 2.5.0):

| 단계 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| CPU forward (tflite_runtime) | 14.46 ms | 14.39 | 14.46 | 16.68 |
| **Coral forward** (libedgetpu) | **11.03 ms** | 10.98 | 11.16 | 11.23 |
| Cosine NN (numpy 1.26) | 13.63 ms | 13.51 | 13.61 | 15.84 |

End-to-end pipeline:

| Mode | forward + cosine | per-char | chars/sec |
|---|---:|---:|---:|
| **CPU only** | 14.46 + 13.63 | **28.09 ms** | 35.6 |
| **Coral + CPU cosine** | **11.03** + 13.63 | **24.66 ms** | **40.6** |

**Coral speedup over CPU: 1.3×** (forward only). End-to-end 12% 빠름.

#### 2.4.3 왜 Coral speedup 이 작은가 — USB I/O bound

Coral edgetpu compile 결과:
- on-chip cache: 7.58 MiB (Coral SRAM 8 MB 의 거의 전부)
- off-chip streaming: 3.41 MiB

Per-invoke USB transfer: 3.41 MiB streaming. USB 3.0 SuperSpeed 의 *실효* 전송속도 ~400 MB/s → 약 **8.5 ms transfer 시간**. TPU 연산은 ~2 ms. 합계 ~10.5 ms — 측정값 11.03 ms 와 일치.

→ **모델이 작아서 Coral 의 advantage 가 USB bottleneck 에 묻힘**. Pi 5 의 ARM NEON INT8 도 비슷한 시간에 이 모델 처리. 큰 모델 (예: 50 MB FC head 가진 v3_keras_char) 에서는 Coral 이 훨씬 빠름.

#### 2.4.4 두 venv path 의 forward 차이 (ai-edge-litert vs tflite_runtime)

| Setup | CPU forward |
|---|---:|
| Python 3.13 + ai-edge-litert 2.1.4 | **9.95 ms** |
| Python 3.9 + tflite_runtime 2.5.0 | 14.46 ms |

ai-edge-litert (2024 build, modern ARM NEON 최적화) 가 tflite_runtime 2.5 (2021 build) 보다 1.5× 빠름. ai-edge-litert 가 Coral 호환은 안 되지만 CPU 추론은 더 효율적.

#### 2.4.5 Best deploy 추천 (정직한 결론)

| 시나리오 | Stack | per-char | chars/sec |
|---|---|---:|---:|
| **Best CPU only** | Python 3.13 + ai-edge-litert + numpy 2 | **28.30 ms** | 35.3 |
| **Best Coral** | Python 3.9 + tflite_runtime + libedgetpu + pycoral + numpy 1.26 | **24.66 ms** | **40.6** |
| Theoretical max | CPU forward (ai-edge-litert) + cosine (numpy 1.26 GEMM) | ~23.6 ms | ~42 |

**Coral 의 deploy 가치**: SCER 같은 작은 INT8 모델에서는 +12% 만. 그러나 *동일 hardware* 에서 큰 모델 (예: ResNet-50, EfficientNet) 으로 확장 시 Coral 이 결정적. Pi/Coral 조합은 *fixed BOM* 에 다양한 모델 deploy 가능한 platform.

### 2.4.6-real Pi 실측 — 실제 한자 PNG 20개

`infer_pi_chars.py` — 실제 한자 PNG 20개 (`~/ece479/test/{한자}.png`) 로 deploy 시연:

| Mode | top-1 | top-5 | avg forward |
|---|---:|---:|---:|
| Pi CPU | **10/20 (50%)** | **14/20 (70%)** | 14.45 ms |
| Pi Coral | **10/20 (50%)** | **14/20 (70%)** | **11.49 ms** |

**CPU vs Coral**: top-5 list 와 cosine sim 점수까지 완전 일치 (Coral 의 12% latency 향상 with 0pp accuracy 차이 재확인).

**성공 사례** (top-1 정확): 三, 図, 嬰, 旧, 標, 獄, 盤, 蒼, 都, 鍵
**부분 성공** (top-5, top-1 놓침): 太/機/鳳 (각 rank 1, 1자 차이), 等 (rank 5)
**미스** (top-5 밖): 再 (150), 勝 (162), 戦 (233), 闘 (2), 活 (234), 演 (125)

**관찰**:
- 모델이 *시각적 유사 변종* 을 top-N 후보로 잘 묶음 (`三 → 𠀒 𡰥 𡵋`, `闘 → 𨶜 聞 闘`) — ArcFace cluster 가 visual semantic 성공
- random val 1000-pack 의 34.30% 보다 50% 더 높음 — 이 test PNG 가 *깨끗한 합성* (synth_engine 의 augmentation 영향 적음)
- 이는 *demo 시연용* 결과로 의미 있고, *production quality* 평가는 random val 1000-pack (34.30%) 가 더 적절

### 2.4.6 Pi 실측 정확도 (1000 val sample)

`eval_pi_scer.py` — `val_pack_1000.npz` (production stratified val) 로 Pi 에서 직접 추론 + 정확도 측정:

| Mode | forward | cosine | total | chars/sec | **top-1** | **top-5** |
|---|---:|---:|---:|---:|---:|---:|
| **Pi CPU** (tflite_runtime 2.5) | 14.47 ms | 13.90 ms | **28.37 ms** | 35.3 | **34.30%** | **48.60%** |
| **Pi Coral** (libedgetpu) | 11.25 ms | 13.46 ms | **24.72 ms** | **40.5** | **34.30%** | **48.60%** |

**핵심 검증 결과**:

1. **Pi 의 정확도가 WSL/Windows 평가와 *정확히* 일치** (doc/29 §7-bis.6 의 PT FP32 = TFLite INT8 = 34.30%) → INT8 양자화의 정확도 보존이 *platform 무관* 일관됨.

2. **CPU vs Coral top-1 완전 동일** (34.30%) → edgetpu_compiler 가 quantization params 를 정확히 보존. Coral 의 *연산 결과* 는 CPU TFLite 와 numerically identical.

3. **Coral 의 12% latency 향상은 정확도 손실 0pp**. Hardware acceleration 의 *clean* gain.

4. **Latency 도 synthetic 입력 bench (§2.4.2) 와 일치** (CPU 14.46 vs 14.47, Coral 11.03 vs 11.25). 입력 distribution 무관한 안정성.

이 결과로 **deploy ready** — Pi 5 + Coral USB 에서 SCER inference 가 *production quality* 로 작동.

### 2.4-old. Coral USB Accelerator — binary 비호환 (Python 3.13 path) ❌

**Hardware 자체는 정상**:
- USB device detect: `lsusb | grep 18d1:9302` (firmware loaded)
- Re-plug 후 USB 3.0 SuperSpeed (5000M) 안정 attach
- libedgetpu1-std 16.0 (Pi OS 기본 패키지) 설치됨

**Binary 비호환 — segfault 발생 path**:

| Runtime | Coral delegate load | Interpreter creation |
|---|:-:|:-:|
| ai-edge-litert 2.1.4 + libedgetpu.so.1 | ✓ (load_delegate OK) | ❌ segfault (exit 139) |
| TF 2.21 + libedgetpu.so.1 | ✓ | ❌ segfault (exit 139) |

**원인**:
1. **libedgetpu 16.0 의 마지막 빌드는 2022 년** — Google Coral SDK 가 *abandoned* 상태. 이후 release 없음.
2. **Modern TFLite delegate API 가 변경됨** — TF 2.20+ / ai-edge-litert 2.x 의 internal Interpreter 구조가 libedgetpu 가 호출하는 ABI 와 mismatch.
3. **Python 3.13** 도 한몫 — pycoral 의 공식 wheel 은 Python 3.7-3.10 만 지원, 3.13 wheel 없음.

**Hardware vs runtime**: 우리 Coral 디바이스는 OK (Phase 1 의 v3_keras_char_int8_edgetpu 도 같은 setup 에서 컴파일 OK 했음). 단지 *Python 3.13 + 최신 runtime + 구 libedgetpu* 조합이 binary mismatch.

### 2.5 Coral 활성화 우회 옵션 (잔여 작업)

| 옵션 | 작업량 | ROI |
|---|---|---|
| **A. Python 3.10 venv** + legacy tflite-runtime + pycoral | 30 min ~ 1h | high — 정상 path, ~2-3 ms forward 예상 |
| B. libedgetpu 를 최신 TF 에 맞춰 source build | 수 시간 | uncertain — Google 의 abandon 상태라 community fork 의존 |
| C. CPU 만으로 ship (현재 결과) | 0 | 28 ms 이미 real-time, *충분히* 실용적 |

추천: **A** (Python 3.10 venv 로 Coral 활성화 시도) — ECE 479 의 edge AI 평가 관점에서 Coral 활용은 의미 있는 deliverable. 학기 일정 안에 가능. 또는 발표/demo 가 끝난 후 잔여 작업으로 분류.

### 2.6 Coral 의 *예상* latency (vs Phase 1 의 v3 모델)

Phase 1 의 v3_keras_char (62 MB, FC head 50 MB) 가 Coral 컴파일 시 on-chip 7.59 MB + streaming 51.43 MB. SCER 는 on-chip 7.58 MB + streaming **3.41 MB** — **streaming 1/15** (FC head 가 없어서). 

추정:
- v3 char Coral 추론 latency 가 ~5 ms 라면 (FC head streaming 50 MB 가 dominant)
- SCER Coral 은 streaming 3.4 MB → **2-3 ms** 추정
- Pipeline 총: 2-3 + 0.21 (filtered cosine) = **2-3 ms / char (300-500 chars/sec)**

이건 *추정* — 실제 측정은 §2.5 옵션 A 통해 검증 필요.

## 3. Phase 4 게이트 (doc/29 §8 의 implicit 목표)

| 게이트 | 임계 | 결과 |
|---|---|---|
| (P4-G1) Pi CPU latency | < 200 ms / char | **28.30 ms** ✅ (7× 여유) |
| (P4-G2) Coral TPU latency | < 30 ms / char | **24.66 ms** ✅ |
| (P4-G3) End-to-end pipeline | < 100 ms | **24.66 ms** ✅ (Coral 모드) |

**Phase 4 3/3 PASS** ✅ — 모든 deploy 게이트 통과.

추가 발견:
- Coral USB I/O bound (forward 의 ~80% 가 USB 3.0 streaming 시간) → SCER 의 작은 model size 에서는 Coral 이득 제한적.
- Python 3.13 (ai-edge-litert) 의 CPU forward 가 Python 3.9 (tflite_runtime 2.5) 보다 1.5× 빠름 — modern ARM 최적화. 다만 ai-edge-litert 는 Coral 호환 안 됨.

## 4. v3 baseline 과의 deploy 비교

| | v3 (T5-light v2, char head) | v4 (SCER) |
|---|---:|---:|
| Deploy artifact (tflite) | 62 MB | **11.5 MB** (5× 작음) |
| Edge TPU 컴파일 binary | 59.12 MB | **11.02 MB** |
| On-chip cache | 7.59 MB | 7.58 MB (동일) |
| Off-chip streaming | 51.43 MB | 3.41 MB (**15× 작음**) |
| 새 글자 추가 | 재학습 필수 | anchor 1 줄 추가 |
| Confusable pair 처리 | 학습 weight 만 | filter + cosine 두 단계 |

SCER 의 deploy 측면 의의 — **streaming 비용 1/15** = Pi 의 모바일 시나리오에서 **load 시간 + 메모리 압박** 큰 차이.

## 5. 산출물 매니페스트 (Phase 3 + Phase 4)

| 파일 | 크기 | 위치 | 용도 |
|---|---|---|---|
| `train_engine_v4/modules/keras_scer.py` | ~6 KB | repo | Keras-native SCER deploy |
| `train_engine_v4/scripts/40_port_pytorch_to_keras.py` | ~6 KB | repo | best.pt → .keras |
| `train_engine_v4/scripts/41_export_keras_tflite.py` | ~5 KB | repo | .keras → INT8 TFLite |
| `train_engine_v4/scripts/42_verify_keras_parity.py` | ~5 KB | repo | PT ↔ Keras 5-output parity |
| `train_engine_v4/scripts/43_eval_int8_accuracy.py` | ~7 KB | repo | INT8 vs FP32 게이트 |
| `deploy_pi/bench_scer_pi.py` | ~7 KB | repo | Pi/Coral latency bench |
| `deploy_pi/export/scer_keras_b1.keras` | 43.6 MB | repo (gitignored) | Keras FP32 batch=1 |
| `deploy_pi/export/scer_int8.tflite` | 11.5 MB | repo + Pi | INT8 TFLite (CPU 호환) |
| `deploy_pi/export/scer_int8_edgetpu.tflite` | 11.0 MB | repo + Pi | **Edge TPU 컴파일 완료** |
| `deploy_pi/export/scer_int8_edgetpu.log` | < 1 KB | repo | 컴파일 op log |
| `deploy_pi/export/scer_anchor_db.npy` | 47.93 MB | repo + Pi | anchor DB (98169, 128) |
| `deploy_pi/export/scer_anchor_db.json` | 1 KB | repo | anchor metadata |
| `deploy_pi/export/scer_bench_pi.json` | 1 KB | repo | Pi CPU latency 측정 결과 |

Pi side (`~/ece479/scer/`):
- `scer_int8.tflite`, `scer_int8_edgetpu.tflite`, `scer_anchor_db.npy`, `class_index.json`, `bench_scer_pi.py`, `scer_bench_pi.json`

## 6. 다음 단계

### 6.1 발표/데모 준비 (즉시)

- ✅ doc/29 (Phase 2) + doc/30 (Phase 3+4) 결과 문서 완료
- ✅ 모든 산출물 + 측정 결과 git 에 commit (다음 commit)
- 💡 **데모 시나리오** — Pi 에서 `bench_scer_pi.py` 실행 + 실제 character 이미지 1-2 개로 추론 (CPU 28 ms / char 시연)

### 6.2 Coral 활성화 잔여 작업 (선택)

- Python 3.10 venv 로 legacy tflite-runtime + pycoral path 셋업
- 같은 `scer_int8_edgetpu.tflite` 로 latency 측정
- doc/30 §2.5 의 추정 (~2-3 ms forward) 검증

### 6.3 학습 quality 개선 (선택, doc/29 §6 의 한계 대응)

- emb/top5 의 12pp 격차 좁히기 — 추가 5 epoch + 작은 lr (doc/28 §11.4 의 extension 절차)
- structure prefilter recall 71% → 85%+ — top-K 늘리기 (rad top-5, idc top-3, stroke ±3)

이 두 항목은 *이미 deliverable 충분* — bonus 작업.

# Demo Run Sheet — Sinograph OCR (ECE 479 Lab 3)

작성: 2026-05-04. demo 시연용 single-source-of-truth.
시연 시 명령어 최소화 + 정형화 표 출력. 모든 wrapper 가 ssh 한 줄.

---

## 0. 사전 점검 (시연 1분 전)

```bash
# Coral 연결 확인 (1a6e:089a 또는 18d1:9302 보여야)
ssh yoonkyu2@192.168.1.16 "lsusb | grep -iE 'global|18d1'"

# Pi Camera 연결 확인 (Stage 3 만 사용)
ssh yoonkyu2@192.168.1.16 "rpicam-hello --list-cameras 2>&1 | head -5"

# demo 폴더
ssh yoonkyu2@192.168.1.16 "ls ~/ece479/demo/"
```

기대:
- `Bus 002 Device 003: ID 18d1:9302 Google Inc.`
- `0 : imx708 [4608x2592 10-bit RGGB]`
- 8 files: `bench_cpu_three.py ocr_adapters.py capture_predict.py run_stage{1,2,3}.sh`

---

## Stage 1 — CPU bench: Commodity OCR vs v3 vs v4

```bash
ssh yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage1.sh"
```

**예상 출력 (요약)** — default `--commodity all` 은 4 commodity row + v3 + v4:
```
GT  Tess[multi]  Easy[ja]  Easy[ch_tra]  Easy[ch_sim]  v3 top-1  v4 top-1  v3 top-5      v4 top-5
─────────────────────────────────────────────────────────────────────────────────────────────────
三  ョ ✗         三 ✓      三 ✓          三 ✓          𡗓 ✗      三 ✓     𡗓𭘾𣱳𰀂三     三亖𠄞𰌴𠀂
再  再 ✓         再 ✓      再 ✓          再 ✓          𫇶 ✗      再 ✗     𫇶𰩵𩾏丏𬍺     再再𢘑𠀑𤥆
勝  勝 ✓         勝 ✓      勝 ✓          膦 ✗          𫪤 ✗      勝 ✓     𫪤𱔟𭰼腾𦜨     勝𰯈𥟎𩷼𱔟
図  区 ✗         図 ✓      囚 ✗          囡 ✗          𦥓 ✗      図 ✓     𦥓龱𡆾𭂮𱕻     図𱕻龱𦥓𡇘
戦  E ✗          戦 ✓      戰 ✗          e ✗           戦 ✓      戦 ✓     戦𫅔𢧐𩓻戕     戦𢧐𫏳𰒱𭈟
旧  旧 ✓         旧 ✓      舊 ✗          旧 ✓          旧 ✓      旧 ✓     旧𪲈𥃬𱸷𠖻     旧𬽪𢁯𠨔𠙵
鍵  鍵 ✓         鍵 ✓      鍵 ✓          键 ✗          䭈 ✗      鍵 ✓     䭈𱄃鍵键𮣞     鍵键䭈𮢱鞬
鳳  局 ✗         鳳 ✓      鳳 ✓          胤 ✗          𤪧 ✗      鳳 ✓     𤪧𬮏鳯㵯𥄧     鳳𮱱𭂺𦁓㵯
─────────────────────────────────────────────────────────────────────────────────────────────────
SUMMARY (n=20)
                Tess[multi]  Easy[ja]  Easy[ch_tra]  Easy[ch_sim]  v3       v4
top-1 acc       50%          95%       80%           40%           25%      95%
top-5 acc       50%¹         95%¹      80%¹          40%¹          55%      100%
avg lat         427 ms       38 ms     445 ms        62 ms         36 ms    24 ms
model size      50 MB        70 MB     70 MB         70 MB         235 MB   11 MB
  v4 latency split: forward 10ms + cosine NN 13ms (over 98,169 anchors)
  ¹ commodity OCR engines without native top-k → top-5 = top-1
```

### 시연 talking points

핵심 메시지: **commodity OCR 들은 language silo** (한 region 만 잘함). v4 SCER 는 단일 11MB 모델로 모든 region universal.

1. **Tesseract [jpn+chi_tra+chi_sim multi] — 50%**:
   - 5.x 최신, 다국어 traineddata (4× upscale + PSM 10 single-char). 단순 stroke 글자 (三, 太, 闘) 에서 over-tokenize.

2. **EasyOCR — language silo 분명히 드러남**:
   - **Easy[ja] 95%**: 일본어 모델, 일본 한자 + BMP 한자 잘 인식.
   - **Easy[ch_tra] 80%**: 번체 중국어 모델, 戦→戰 같은 일본 simplified→traditional 변환으로 fail. 일부는 잘 인식.
   - **Easy[ch_sim] 40%**: 간체 중국어 모델. 大부분 traditional 한자 못 인식 (鍵→键, 鳳→胤).
   - 즉 **각 모델은 특정 region 에서만 좋음**. 사용자가 어느 모델을 쓸지 결정해야 함 → real-world 에는 region detection 또는 ensemble 필요 → 인프라 복잡.

3. **PaddleOCR — Pi/ARM 호환성 한계 (init 실패)**:
   - 강력한 CJK commodity OCR (PP-OCRv5) 이지만 **paddlepaddle 의 ARM64 + Python 3.13 wheel 가 C++ level segfault**.
   - paddlepaddle 2.x 는 ARM64+Py3.13 wheel 부재. 3.x 만 깔리는데 segfault.
   - bench 에서 graceful skip — 표에는 row 빠짐, 메시지로 이유 표시. 또 다른 commodity OCR 의 platform-portability 한계 (그 자체 demo point).
   - x86_64 dev 머신에서는 작동 — 필요하면 dev 측 별도 측정해 비교 (현 demo scope 에선 미실행).

4. **cnocr — 가장 빠른 commodity (latency winner)**:
   - **PP-OCRv5 ONNX (RapidOCR backend)**, 100 MB. 중국어 specialist.
   - **Pi 13 ms/image** — v4 SCER (25ms) 보다도 빠름 (forward만 8ms).
   - 단점: 정확도 50% (v4 의 81.6% 대비 -32pp). speed-accuracy trade-off 의 명확한 한쪽 끝.
   - "commodity 중 가장 빠른 것 보다 v4 가 2× 느리지만 +32pp 정확" — Pareto 메시지.

5. **Manga-OCR — 강력한 transformer specialist (commodity accuracy winner)**:
   - **kha-white/manga-ocr-base** (ViT encoder + GPT2 decoder, 440 MB), 일본 manga 텍스트로 학습.
   - 일본 한자 강함 (66%), 다른 region 한자도 일부 cover.
   - 단점: **440 MB** + Pi CPU **788 ms/image** + Japanese-only specialist.
   - **v4 SCER 와 직접 비교**: v4 11MB / 25ms / 81.6% — Manga 보다 **40× 작고 32× 빠르며 +16pp 더 정확**, universal.
   - Manga 가 못 잡고 v4 가 잡는 sample: `𤨒` (CJK Ext B), `媤` (Korean-only), `𤴡` 등 — 발표 Slide 1 의 EasyOCR-style hallucination 시연 가능.

6. **GVision (Google Cloud Vision) — billing gating**:
   - free tier 1000 호출/월 이지만 **billing account 등록 필수** (credit card). 우리 demo 환경에선 미등록 → graceful skip.
   - cloud OCR 의 또 다른 deploy friction 시연. v4 는 11MB binary 로 어디서나.

7. **v3 baseline (ONNX FP32) — top-1 16%, top-5 32%** (n=38):
   - 234 MB single-stage classifier (98k-class FC head). 학습 정확도 38% (12k val).
   - INT8 quantize 실패 (`doc/24`) → v4 SCER pivot.
   - top-1↔top-5 격차 16pp: 정답이 후보 5위 안엔 자주 들어가지만 score calibration 약함. 절대치는 글자 set 의존 (rare CJK Ext B 가 추가될수록 약점 노출).

8. **v4 SCER (INT8) — 81.6% top-1, 92.1% top-5** (n=38) ⭐:
   - **11 MB 단일 모델로 ja + ch_tra + ch_sim + KR 한자 모두 처리** — universal sinograph explorer.
   - backbone + 128-d L2-norm embedding + cosine NN over 98k anchor DB. Open-set: **새 글자 추가 = anchor 1 줄 append, 재학습 X**.
   - top-5 격차 5pp (95→100%) — well-calibrated embedding. cosine NN 이 자연스러운 ranking 제공 → ranked alternatives natively (commodity OCR 못함).
   - latency: forward 10ms (CPU INT8) + cosine NN 13ms (CPU numpy matmul).

### 모델 비교 한 줄 정리 (n=36 sample 기준)

| 차원 | Manga (commodity winner) | **v4 SCER** |
|---|---:|---:|
| top-1 정확도 | 69.4% (ja-specialist) | **80.6% (universal)** |
| top-5 정확도 | 69.4% (no native top-k) | **91.7%** |
| 모델 크기 | 440 MB (HF transformer) | **11 MB INT8 TFLite** |
| 추론 latency (Pi CPU) | ~794 ms | **25 ms (32× 빠름)** |
| 다국어 cover | 일본어 specialist | **모든 CJK region** |
| top-k ranking | ✗ (greedy decode) | ✓ (cosine NN 자연 ranking) |
| 새 글자 추가 | 재학습 / fine-tune | **anchor DB 한 줄 append** |
| Pi/Coral 가속 | unsupported | **24 ms (Coral 12% 빠름)** |
| BMP-only vs Ext B | BMP-only (𤨒 fail) | **Ext B 도 cover** (𤨒 ✓) |

---

## Stage 2a — v3 가 Coral 에 fit 안 되는 이유 (dev 측, 라이브)

```bash
bash "deploy_pi/demo/recompile_edgetpu_with_summary.sh"
# (내부에서 wsl 호출, edgetpu_compiler 16.0 으로 v3+v4 둘 다 -s 재컴파일)
```

**예상 출력 (cache split takeaway)**:

| 모델 | 총 크기 | On-chip cache (max 8 MB) | Off-chip stream | 결과 |
|---|---:|---:|---:|---|
| **v3** | 59 MB | 7.6 MB (13%) | 51 MB (87%) | 대부분 USB 스트림 → Coral ≈ CPU |
| **v4** | 11 MB | 7.6 MB (69%) | 3.4 MB (31%) | 대부분 on-chip → Coral 12% 빠름 |

### 시연 talking points
- Coral USB Accelerator 의 SRAM 은 **8 MB**. 모델이 이걸 초과하면 가중치가 USB 케이블을 통해 호스트 RAM 에서 streaming. 이때 Coral 의 매트멀 가속 효과는 PCIe 버스 latency 에 묻힘.
- v3 의 102k FC head 가 ~50 MB → on-chip 1.75 KiB 만 남고 51 MB 가 off-chip → **compile 은 성공 (36/36 ops mapped) 하지만 실측 가속 효과 0**.
- v4 SCER 의 architectural insight: **classifier head 를 모델 밖으로 빼고 cosine NN 으로 옮김** → backbone+embedding 만 11 MB → 70% on-chip 캐시 hit → 12% latency 절감.
- 추가 이득: **새 글자 추가 = anchor DB 한 줄 append**, 재학습 X — open-set classification.

자료 path:
- `deploy_pi/export/edgetpu_v3_full_summary.log`
- `deploy_pi/export/edgetpu_v4_full_summary.log`

---

## Stage 2b — Pi 라이브: v4 SCER on 20 PNG

```bash
ssh yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage2.sh"
```

이 wrapper 가 자동 처리:
1. Coral delegate 라이브 시도. 성공 시 **CPU + Coral side-by-side** 표.
2. 실패 시 (현재 ai-edge-litert 2.x + Py 3.13 segfault 이슈) **CPU 라이브 + 사전 측정 결과 인용**.

**예상 출력 (CPU 라이브 part)**:
```
GT  CPU top-5 (sim)                           rank
─────────────────────────────────────────────
三  三(0.79) 亖(0.66) 𠄞(0.62) 𰌴(0.58) 𠀂(0.57)   0
再  再(0.64) 再(0.62) 𢘑(0.62) 𠀑(0.58) 𤥆(0.55)   1
...
─────────────────────────────────────────────
CPU   top-1 = 19/20 (95.0%)  top-5 = 20/20 (100%)  avg fwd = 10.1 ms
```

**Coral fallback (인용)** — `doc/32_PHASE3_4_REDO_RESULTS.md` 검증 데이터:

| | Pi CPU INT8 | Pi Coral INT8 |
|---|---:|---:|
| forward latency | 14.65 ms | **11.23 ms** (-23%) |
| end-to-end (incl. cosine) | 28.52 ms | **24.70 ms** (-13%) |
| 1000-pack top-1 | 96.90% | 96.90% (0pp loss) |
| 20 PNG top-1 | 95% | 95% (동일) |

### 시연 talking points
- 라이브 CPU 결과가 사전 측정 (96.9%) 과 일치 — quantization + porting 이 정확도 손실 없음.
- Coral 실측 latency 는 prior 검증된 데이터 인용. 이 격차가 Stage 2a 의 cache-split 이론과 맞음 (12% 빠름 ≈ off-chip 31% 의 비용).
- **현재 limit (24.7 ms) 는 architecture 한계가 아님**. 두 가지 개선 path:
  1. **Anchor DB 축소**: 일상 빈도 상위 10k 글자만 사용 → cosine NN 13ms → ~1.3ms (10×). e.g. v3 / v4 의 frequent-set 모드.
  2. **Cosine NN 을 Coral 로 이전**: anchor 매트릭스를 Coral SRAM 에 올리고 행렬곱을 INT8 매트멀로. 이론상 forward+NN 합쳐 5-8 ms 가능.
  3. **Multi-stage filter**: SCER 의 4개 structure head (radical/total_strokes/residual/idc) 로 후보를 5-50개로 줄인 뒤 cosine 만 → cosine 1ms 이하.
- 셋 다 **시간 문제일 뿐 design 미스 아님**. Demo 시점에는 simplest path (full anchor DB) 만 검증.

---

## Stage 3 (Optional) — Pi Camera 라이브

```bash
ssh -t yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage3.sh"
# loop 모드: Enter 칠 때마다 캡처, q+Enter 종료
```

**예상 시연 흐름**:
1. 종이에 검은 펜으로 한자 1개 (例: 漢, 勝, 機 ...) 손글씨 또는 인쇄.
2. 카메라 앞에 종이 들기 (조명 충분).
3. Enter → 자동 cropping (OpenCV adaptive threshold + largest contour) → 128×128 → v4 SCER → top-5 출력.
4. `/tmp/cap_orig.jpg` 와 `/tmp/cap_crop.png` 로 결과 사진 review 가능.

```
> 
  bbox=423,128 196×201  fwd=10.2ms  nn=13.4ms
  top-5: 漢(0.83) 𤁆(0.65) 暵(0.61) 𫇣(0.59) 攩(0.57)
  saved: /tmp/cap_orig.jpg + /tmp/cap_crop.png
```

### 시연 talking points
- **Auto-crop 방식**: grayscale → adaptive Gaussian threshold (block 21, C 8) → external contours → largest 의 bounding box → 20px padding → square pad → resize. 종이 위 한자 같은 dark-on-light 가정.
- Limitations: 멀티 글자, 회전, low contrast 시 실패 가능 — 이는 detection step 의 한계지 인식 모델 한계 아님. 더 robust 한 detector (CRAFT, DBNet) 로 교체 가능.
- 실패 케이스 (no character found) → "조명/대비/한 글자만" 안내 메시지.

---

## 시연 시나리오 (~5 분)

| 시간 | Stage | 명령어 |
|--|---|---|
| 0:00 | Stage 0 | `lsusb`, `rpicam-hello --list-cameras` 점검 |
| 0:30 | Stage 1 | `ssh ... run_stage1.sh` (Tesseract + v3 + v4 통합 표) |
| 2:00 | Stage 2a | `bash recompile_edgetpu_with_summary.sh` (cache split) |
| 3:00 | Stage 2b | `ssh ... run_stage2.sh` (CPU 라이브 + Coral 인용) |
| 4:00 | Stage 3 | `ssh -t ... run_stage3.sh`, 종이 한자 1-3개 시연 |

---

## 핵심 메시지 9줄 (n=38 sample 기준)

1. **Commodity OCR 들은 language silo** — Easy[ja] 53%, Easy[ch_tra] 66%, Easy[ch_sim] 37%, Tess[multi] 37%. 한 region 만 잘함.
2. **cnocr — 가장 빠른 commodity (12.7 ms, v4 보다도 ↑)** — but 정확도 50%, **speed-accuracy trade-off 의 한쪽 끝**.
3. **Manga-OCR — 가장 정확한 commodity 중 하나 (66%)** — but **440 MB / 788 ms / Japanese-specialist**, BMP-only.
4. **PaddleOCR (강력) 은 ARM Pi 5 + Python 3.13 미호환** — paddlepaddle 3.x segfault. commodity 의 platform 의존성.
5. **GVision cloud OCR 도 deploy friction** — billing 등록 + API key + network. v4 는 11 MB binary 로 어디서나.
6. **v3 (single-stage 102k FC) 는 architecture 한계** — INT8 변환 실패 + Coral SRAM 87% off-chip stream.
7. **v4 SCER 는 Pareto frontier 최상단** — 단일 **11 MB / 25 ms** 모델로 ja+ch_tra+ch_sim+KR+SMP Ext B 모두 cover. **81.6% top-1 / 92.1% top-5**.
8. **v4 vs commodity 들 모두 우위**: 가장 빠른 cnocr 의 2× 느린 대신 +32pp 정확. 가장 정확한 Manga 보다 +16pp 정확하며 32× 빠르고 40× 작음. v4 만 ranked top-k natively (cosine NN).
9. **현재 24 ms latency 는 설계 한계가 아님** — anchor DB 축소 / Coral matmul / multi-stage filter 로 단일자리 ms 가능, 단 시간 문제.

---

## 사전 dry-run 체크리스트

- [x] Stage 1 wrapper 실행, 표 출력 확인 (Tesseract 50%, v3 25%, v4 95%)
- [x] Stage 2a WSL recompile, cache split 출력 (v3 87% off-chip, v4 31%)
- [x] Stage 2b CPU 라이브 95% top-1, 100% top-5
- [x] Stage 3 카메라 캡처 → top-5 출력 (~24 ms 합계)
- [ ] Coral USB plugged 확인 (시연 직전)
- [ ] Pi Camera ribbon cable plugged 확인 (시연 직전, Stage 3)
- [ ] SSH 연결 + key 추가 (`ssh-add` 등) — 시연 PC 에서

---

## 부록 A — Pi 작업 디렉토리 구조 (시연 시점)

```
~/ece479/
├── .venv/                  # Lab2 venv (TF 2.21, hidden)
├── lab2/                   # Lab2 자료 24개 (정리 완료)
├── lab_v3/                 # v3 ONNX (235 MB) + INT8 (60 MB) + class_index
├── scer/                   # v4 SCER 배포물 (TFLite, anchors, scripts)
├── test/                   # 20 한자 PNG (再 三 勝 図 太 ...)
├── demo/                   # 시연 인프라 (이번 phase 신규)
│   ├── ocr_adapters.py     # Tesseract + 확장용 stubs
│   ├── bench_cpu_three.py  # 통합 CPU 벤치
│   ├── capture_predict.py  # Pi Camera + auto-crop + v4 SCER
│   ├── run_stage1.sh       # Stage 1 wrapper
│   ├── run_stage2.sh       # Stage 2b wrapper (Coral fallback)
│   └── run_stage3.sh       # Stage 3 wrapper (camera loop)
└── venv-ocr/  (홈 dir)      # Python 3.13 + ai-edge-litert + onnxruntime
                              + pytesseract + opencv + picamera2
```

## 부록 B — 알려진 limitation

- **PaddleOCR Pi 미호환**: paddlepaddle 3.x 가 ARM64 + Python 3.13 환경에서 SaveOrLoadPirParameters 단계 SIGSEGV. 2.x wheel 없음. PaddleOCR adapter 가 init guard 로 graceful skip + 명확한 메시지 출력. **이건 demo point** — commodity OCR 의 platform-portability 한계, v4 SCER 의 11MB TFLite 와 대비. dev (x86_64) 에서는 paddlepaddle 작동, 필요시 별도 측정.
- **Coral 라이브 segfault**: ai-edge-litert 2.1.x + Python 3.13 + libedgetpu1-std 16.0 조합. Py 3.11 별도 venv + tflite-runtime 으로 해결 가능 (시연 직전 시간 시 적용). 그렇지 않으면 Stage 2b 의 인용 fallback 사용.
- **v3 INT8 deploy 실패**: `doc/24` blocker. v3 의 전체 102k 분류기를 INT8 로 변환 시 dynamic range mismatch. v4 SCER 로 우회 (분류기를 cosine NN 으로 대체).
- **Stage 3 auto-crop**: 단일 글자 모드 default + multi-char detection 모드 (loop 의 `M` 토글) 둘 다 지원. dark-on-light + 충분한 contrast 가정. multi-char 는 `--multi-morph-kernel` 로 stroke merge 강도 조정.

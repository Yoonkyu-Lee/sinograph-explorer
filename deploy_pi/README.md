# Pi Deployment — Sinograph OCR

Raspberry Pi 로 본 프로젝트 OCR 모델 배포. Lab2 Part 2 의 `.pth → ONNX → TFLite → Coral TPU` 경로와 동일 구조. 현재는 **스켈레톤 단계** — 실제 배포는 production 학습 완료 후.

## 파이프라인

```
[PC]  .pth best.pth                                   ← Stage 2 학습 출력
   ↓  (train_engine_v2/scripts/22_export_onnx.py)
      .onnx  (FP32, 11 M params)
   ↓  (train_engine_v2/scripts/23_quantize_tflite.py)
      .tflite (INT8, ~17 MB quantized)
   ↓  (edgetpu_compiler, Linux 만)
      .tflite _edgetpu (Coral accelerator compiled)

[Pi]  .tflite 파일 + class_index.json
   ↓  (deploy_pi/infer_pi.py)
   예측: codepoint + 한자 literal + (옵션) canonical family
```

## 현 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| PC `.pth` → `.onnx` | ✅ 완성 (`22_export_onnx.py`) | `--run-dir` 로 best.pth 자동 로드 |
| PC `.onnx` → `.tflite` INT8 | 🟡 스텁만 (`23_quantize_tflite.py`) | onnx2tf + tf-nightly 설치 필요, production 학습 후 구현 |
| PC `.tflite` → Coral compiled | ⚠️ Linux only | edgetpu_compiler 는 Windows 미지원 → WSL 또는 Pi 자체에서 |
| Pi `.tflite` → inference | ✅ 스크립트 완성 (`infer_pi.py`) | tflite_runtime + Coral delegate 지원 |

## Pi 초기 설정

Lab2 의 `ECE479` pyenv 환경 재활용 가능:

```bash
# Pi SSH 후
pyenv activate ECE479          # Lab2 에서 만들었던 거
pip install -r requirements_pi.txt
```

Coral TPU 사용시 추가:
```bash
sudo apt-get install libedgetpu1-std
# 또는 max (발열 주의, 성능 ↑): sudo apt-get install libedgetpu1-max
```

## 파일 전송 (PC → Pi)

scp 또는 USB. 필요한 파일:
- `model_int8.tflite` (~17 MB) 또는 `model_int8_edgetpu.tflite` (Coral 용)
- `class_index.json` (~200 KB, 10,932 entries)
- `infer_pi.py` (이 폴더의 스크립트)
- 테스트 이미지 (`.jpg` / `.png` 단일 글자 crop)

선택:
- `canonical_v2.sqlite` (~50 MB, variant family 조회용)

## 실행 (Pi 쪽)

```bash
# CPU inference (모든 Pi 에서 가능)
python3 infer_pi.py \
  --model model_int8.tflite \
  --class-index class_index.json \
  --image my_photo.jpg \
  --topk 5

# Coral TPU inference (Coral dongle 연결시)
python3 infer_pi.py \
  --model model_int8_edgetpu.tflite \
  --class-index class_index.json \
  --image my_photo.jpg \
  --use-coral \
  --topk 5

# + canonical DB family lookup
python3 infer_pi.py --model model_int8.tflite --class-index class_index.json \
  --image my_photo.jpg --family-db canonical_v2.sqlite --topk 5
```

## 출력 예시

```
[infer] model: model_int8.tflite
[infer] input: shape=[1, 3, 128, 128] dtype=int8
[infer] output: shape=[1, 10932] dtype=int8
[infer] classes: 10932
[infer] coral: False

[timing] preprocess=4.2ms  invoke=180.5ms  total=184.7ms

=== Top-5 predictions ===
  #1  U+4E2D  '中'  prob=87.3%
  #2  U+4E9C  '亜'  prob=2.1%
  ...

[family] top-1 U+4E2D variant family (2): 中 塚
```

## 예상 성능 (ResNet-18, 128², INT8)

| 환경 | 추론 시간/이미지 | 비고 |
|---|---|---|
| Pi 4 CPU | 150~400 ms | 단독 CPU, 충분히 실시간 |
| Pi 5 CPU | 80~200 ms | 2~3× 빠름 |
| Pi + Coral TPU | **10~30 ms** | Edge TPU offload, 하지만 edgetpu_compile 성공 전제 |
| PC (GPU, 30_predict.py) | 5 ms | 참조 기준 |

## 전처리 일치 (중요)

**`.tflite` 가 재학습 시 사용한 전처리와 정확히 일치해야 함**. 이 프로젝트의 현 파이프라인:

```
PIL.Image.open → RGB
resize shorter side to 128 (bilinear)
center crop to (128, 128)
float / 255            → [0, 1]
(x - 0.5) / 0.5        → [-1, 1]
transpose (H,W,C) → (C,H,W)
add batch dim → (1, 3, 128, 128)
```

`infer_pi.py` 의 `preprocess()` 함수가 위 단계 그대로 구현. 학습 config 바뀌면 이 함수도 같이 업데이트해야 함.

## 알려진 함정

1. **BGR vs RGB**: OpenCV 는 BGR 기본. 우리는 PIL 로 RGB 로 로딩해서 일관성 유지. OpenCV 쓸 경우 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 필수.
2. **Quantization scale/zero_point**: INT8 tflite 는 input 이 int8 이라 전처리 결과 float 을 quantize 해서 넣어야 함. `infer_pi.py` 의 `quantize_if_int8()` 가 처리.
3. **Edge TPU op 호환**: edgetpu_compiler 가 일부 op 를 TPU 에 매핑 못 하면 CPU fallback 발생 → 느려짐. 컴파일 로그 확인 필수.
4. **Lab2 경험**: 카메라 (picamera2 / cv2) 와 inference 는 별개 환경에서 돌리는 게 안정적. infer 스크립트는 이미지 파일만 받음 (실시간 카메라 pipeline 은 별도).

## 이후 개발 (production 학습 후)

1. **23_quantize_tflite.py 실제 구현**:
   - `onnx2tf` 로 ONNX → TF SavedModel
   - `tf.lite.TFLiteConverter` 로 INT8 quantization (calibration: shard 에서 300 samples)
   - 출력 `.tflite` FP32 accuracy vs INT8 accuracy 비교 (drop < 2%p 이상이면 OK)
2. **Coral edgetpu_compile**: WSL Ubuntu 또는 Pi 자체에서 `edgetpu_compiler model_int8.tflite`
3. **실측 Pi 배포**: SSH 로 Pi 에 `scp` + `infer_pi.py` 실행, 실제 latency / accuracy 측정
4. **(선택) 실시간 카메라 통합**: picamera2 로 frame grab → crop (manual ROI 또는 간단 detection) → infer_pi → 결과 오버레이

## 참고

- `doc/00_CONTEXT_TILL_LAB2.md` — Lab2 Part 2 / Part 3 의 TFLite / Coral 경험 요약
- `doc/07_TWO_STAGE_WORKFLOW.md` — 전체 2-stage 파이프라인
- `train_engine_v2/MANUAL.md` — PC 쪽 학습 엔진 사용법
- `train_engine_v2/scripts/22_export_onnx.py` — PC ONNX export
- `train_engine_v2/scripts/23_quantize_tflite.py` — TFLite 변환 (스텁)

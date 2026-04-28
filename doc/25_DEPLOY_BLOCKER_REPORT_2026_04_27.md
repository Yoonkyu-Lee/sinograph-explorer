# Edge TPU 배포 차단 진단 보고서

작성: 2026-04-27.  대상: T5-light v2 (98,169 class, 39% top-1) Pi/Coral 배포.

이 문서는 **현재 사용 중인 변환 경로 (onnx2tf / onnx_tf) 가 Edge TPU 매핑을
실패시키는 실험적 차단 요인** 과 그 진단 과정을 정리. Final Demo 에서 TPU vs
CPU 추론 속도 비교가 요구되므로, **현 경로로는 데모 불가, 모델 재구현이
필요함** 을 명시.

> 참고 (2026-04-28 추기): doc/27 Phase 1 에서 "Keras-native 재구현 → TF native
> converter" 경로로 36/36 ops mapped 가 실증되어, doc/25 가 식별한 차단 요인이
> 변환 경로 선택의 문제였음이 확정됨. 단 본 문서가 추정한 "정확한 schema 차이"
> 는 flatbuffer 수준 비교가 아니라 op-level error 메시지에 기반한 추론이며,
> 정확히는 "ONNX-derived 양자화 메타데이터 vs TF native 양자화 메타데이터" 의
> 차이로 정리하는 편이 안전. doc/27 §1 / 메모리 `project_keras_porting_pitfalls`
> 참고.

관련 문서:
- [doc/22](22_TRAIN_GPU_OPTIMIZATION_PLAN.md) — synth GPU 최적화
- [doc/23](23_PHASE_TG1_RESULTS.md) — TG-1 측정
- [doc/24](24_DEPLOY_BLOCKERS_AND_V4_PLAN.md) — 초기 배포 한계 + SCER 제안 (이 문서가 §6 갱신)

---

## 1. 배경 — Final Demo 요구사항

Lab3 Final Demo:
- TPU 가속 vs CPU 추론 latency 비교 필수
- Lab2 의 비교 frame work (`cpu_vs_edge_tpu_latency_benchmark.py`) 와 동일 형식
- 즉 **Edge TPU 매핑 N > 0** 이어야 의미 있는 비교 가능

따라서 Edge TPU 호환성은 데모의 비협상 사항.

---

## 2. 현재 작동 상태

### 2.1 학습된 모델
- `train_engine_v3/out/15_t5_light_v2/best.pt` (epoch 10, 98,169 class)
- val char/top1 = 38.99%, char/top5 = ~52%, radical/top1 = 71.4%, idc/top1 = 94.2%

### 2.2 ONNX FP32 → Pi 추론 (작동 중)
- `deploy_pi/export/v3_char.onnx` (246 MB, opset 17)
- `deploy_pi/export/v3_char_opset11.onnx` (246 MB, opset 11 — Lab2 호환용)
- Pi 추론 (`onnxruntime` 1.x): **36.8 ms / 이미지, 26.3 img/s**
- 실 이미지 20개에서 char-only top-1 = 5/20 (25%)

### 2.3 자산
- `deploy_pi/export/class_index.json` (98,169 entries, v3)
- `deploy_pi/infer_pi_onnx.py` (작동 검증)
- `synth_engine_v3/out/94_production_102k_x200/` (corpus + aux_labels.npz)
- `train_engine_v3/scripts/30_predict.py` (PC 검증, fusion 모드 포함)

---

## 3. Edge TPU 시도 — 모두 실패

### 3.1 시도 1 — `onnx2tf` direct INT8 (full 98k 모델)

```
ONNX → onnx2tf.convert (output_integer_quantized_tflite=True)
   → v3_char_full_integer_quant.tflite (63 MB)
   → edgetpu_compiler
   → 0/39 ops mapped, "unsupported data type"
```

원인 가설: onnx2tf 의 INT8 quantization metadata 형식이 Edge TPU compiler v16
의 spec 과 불일치.

### 3.2 시도 2 — `onnx_tf` + TF native converter (full 98k 모델)

Lab2 의 정통 경로:
```
ONNX → onnx_tf.backend.prepare()
   → SavedModel
   → tf.lite.TFLiteConverter.from_saved_model()
   → INT8 spec (TFLITE_BUILTINS_INT8, inference_input_type=int8)
   → v3_char_int8_lab2_strict.tflite (62 MB)
   → edgetpu_compiler
   → SILENT CRASH (no log file)
```

원인 가설: 50MB final FC 가 컴파일러의 단일 텐서 한계 초과 → silent fail.

### 3.3 시도 3 — Backbone-only (no char head)

50MB FC 제거 후:
```
ONNX (backbone only, 45 MB FP32, 11M params)
   → onnx2tf direct INT8
   → v3_backbone_full_integer_quant.tflite (11 MB)
   → edgetpu_compiler
   → 0/38 ops mapped, "unsupported data type"
```

또는:
```
ONNX (backbone only) → onnx_tf → TF native converter
   → v3_backbone_int8.tflite (11 MB)
   → edgetpu_compiler
   → SILENT CRASH
```

**FC 제거해도 호환 안 됨.** 이게 결정적 발견.

### 3.4 비교 — Lab2 의 작동 모델

같은 toolchain (Edge TPU compiler v16, lab2-style-venv 환경) 로 검증:

| 모델 | 크기 | 매핑 결과 |
|---|---|---|
| `model_4x4_fullint.tflite` (Lab2 FashionNet) | 0.76 MB | **8/8 ops mapped ✓** |
| `facenet_fullint.tflite` (Lab2 Inception ResNet) | 23 MB | **181/181 ops mapped ✓** |
| 우리 v3 backbone (onnx2tf path) | 11 MB | **0/38 ops mapped ❌** |
| 우리 v3 backbone (onnx_tf path) | 11 MB | **silent crash ❌** |

→ **toolchain 정상**, **우리 출력 tflite 만 호환 안 됨**.

---

## 4. 근본 원인 — Op-level 불일치

`onnx2tf` path 의 Edge TPU 컴파일 로그:

```
PADV2          1   "Operation is working on an unsupported data type"
PAD            4   "Operation is working on an unsupported data type"
CONV_2D        20  "Operation is working on an unsupported data type"
ADD            8   "Operation is working on an unsupported data type"
QUANTIZE       1   "unspecified limitation"
DEQUANTIZE     1   "Operation is working on an unsupported data type"
MAX_POOL_2D    1   "Operation is working on an unsupported data type"
MEAN           1   "Operation is working on an unsupported data type"
RESHAPE        1   "Operation is working on an unsupported data type"
```

비교 — FaceNet 의 매핑된 op:
```
CONV_2D          132  Mapped
FULLY_CONNECTED  1    Mapped
ADD              21   Mapped
MAX_POOL_2D      3    Mapped
CONCATENATION    23   Mapped
MEAN             1    Mapped
```

같은 op type 인데 결과 다름. 차이는 **op 의 quantization metadata 형식**:
- Edge TPU 는 specific INT8 schema 기대 (per-axis weights, per-tensor activations,
  특정 scale/zero_point 형식)
- 우리 toolchain (onnx2tf 또는 onnx_tf+TF 2.15 native converter) 가 약간 다른
  schema 출력
- 컴파일러가 **dtype mismatch 로 판단해서 모두 reject**

**결정적 차이점들 (가설):**
1. **Explicit padding op**: PyTorch ResNet-18 의 ONNX export 가 conv 앞에 명시적
   `Pad` op 를 추가. FaceNet (Inception architecture, padding="same") 은 이게
   없음. Edge TPU 가 explicit Pad+Conv 패턴 미지원.
2. **Quantization metadata 버전**: TF 2.15 (2024) 의 TFLite 출력이 Edge TPU
   compiler v16 (2022 마지막 release) 의 schema 와 호환 안 됨.
3. **PyTorch ↔ TF 변환의 op-level 차이**: torchvision ResNet 의 batch_norm
   folding, skip connection ADD 의 quantization 처리가 Keras-native ResNet 과
   다른 패턴 산출.

이 셋이 합쳐서 silent crash 또는 0% mapping 으로 이어짐.

---

## 5. 분석 — 왜 backbone 만 분리해도 안 되나

SCER (doc/24 §6) 의 핵심 약속은:
> 50MB FC 제거 → backbone (11M) 만 deploy → Coral 100% 매핑

이 PoC 가 **거짓** 으로 판명. backbone 만 분리해도 **같은 op-level 비호환** 을
계속 만남. FC 가 문제의 전부가 아니었음.

따라서 SCER 의 architecture 변경 자체로는 Edge TPU 호환성 못 얻음. **모델
변환 path 가 근본 원인** 이지 architecture 가 원인 아님.

---

## 6. 결정 — 모델 재구현 불가피

Final Demo 의 TPU vs CPU 비교 요구를 만족시키려면:

### 6.1 옵션 — 재구현 방식 비교

| 방법 | 설명 | 작업량 | Edge TPU 가능성 | 정확도 영향 |
|---|---|---|---|---|
| **A. Keras-native ResNet-18 재구현** | Pure Keras 로 ResNet 빌드, PyTorch weight 이식 | 3-5일 | 높음 (FaceNet 와 같은 path) | 동일 (가중치 이식) |
| **B. MobileNetV2 / EfficientNet-Lite** 로 backbone 교체 | Edge TPU 검증된 backbone, 처음부터 학습 | 2-3주 | 매우 높음 | 재학습 필요, 결과 미지수 |
| **C. ONNX → TF Keras 자동 변환 후 재학습** | onnx-to-keras 라이브러리 사용 + fine-tune | 1-2주 | 높음 (불확실) | fine-tune 효과 따라 |
| **D. PyTorch + 직접 quantize → manual TFLite** | torch.quantization → custom export | 1주 | 낮음 (실험적) | INT8 손실 큼 |

### 6.2 권장 — 옵션 A (Keras-native 재구현)

**근거:**
- Lab2 의 FaceNet 도 같은 path (Keras → TFLite → Edge TPU) 로 작동 검증
- ResNet-18 은 Keras 로 ~200줄 직접 빌드 가능
- PyTorch weights 를 Keras 로 이식: weight tensor 형식만 transpose (CHW ↔ HWC)
- 학습 다시 안 해도 됨 — `best.pt` 의 weights 그대로 활용
- 그 위에 SCER architecture (small heads + embedding) 결합 가능

**작업 단계:**
1. **Keras ResNet-18 빌드** (~200줄)
   - `tf.keras.layers.Conv2D` + `BatchNormalization` + `ReLU` + skip connections
   - torchvision 의 ResNet-18 layer-by-layer 매칭
2. **PyTorch weight → Keras weight 이식 스크립트** (~150줄)
   - state_dict 의 각 layer mapping
   - Conv weights: PyTorch (out, in, h, w) → Keras (h, w, in, out)
   - BatchNorm: gamma/beta/running_mean/running_var 동일
3. **검증**: 같은 input 에 대해 PyTorch vs Keras 출력 max diff < 1e-4
4. **Keras → SavedModel → INT8 TFLite** (Lab2 FaceNet path 동일)
5. **edgetpu_compile** 매핑 검증 (목표: backbone 100% 매핑)
6. (옵션) SCER architecture 추가 (embedding head, small heads) 후 fine-tune

### 6.3 SCER 와의 관계

옵션 A 가 SCER (doc/24) 와 호환:
- Keras ResNet-18 backbone 위에 SCER 의 small heads 추가
- 학습 fine-tune (warm restart from PyTorch weights)
- Edge TPU 호환성 확보된 상태로 SCER 의 novelty 발휘

즉 **옵션 A 는 SCER 의 prerequisite**.

---

## 7. 시간 계획

| 단계 | 시간 | 산출물 |
|---|---|---|
| Keras ResNet-18 빌드 + PyTorch weight 이식 | 1-2일 | `keras_resnet18.py`, `port_pytorch_to_keras.py` |
| INT8 TFLite + Edge TPU compile 검증 | 0.5일 | mapping 결과 (목표 100%) |
| 실 Pi 측 추론 검증 (CPU vs Coral 속도) | 0.5일 | latency 측정 결과 |
| SCER architecture 추가 + 학습 (warm restart) | 2-3일 | v4 ckpt |
| 발표 자료 + demo 영상 | 1일 | final demo 자산 |

**총 5-7일**. 발표 (다음 주) 까지는 PoC 수준 결과, 데모 (다음 다음 주) 에 완성.

---

## 8. 발표 시점 (다음 주) 의 메시지

발표 시점에 데모 성공 안 해도 **현재 상태로 충분**:
- v3 ONNX FP32 Pi 추론 결과 (25% top-1, 36.8ms)
- 학습 곡선 (10 epoch 의 마지막 cosine 수렴 효과)
- TPU 차단 발견 (intentional learning experience)
- v4 plan (재구현 정당성 + SCER novelty)

발표 핵심: "현 상태 + 발견한 한계 + 해결 plan". 데모는 다음 주차에.

---

## 9. 리스크 / 미해결 질문

1. **Keras weight 이식 정확도**: torchvision ResNet 의 specific impl 과 Keras 표준
   ResNet 사이의 layer 정의 미세 차이 (예: padding, bias 처리). 검증 필수.
2. **TF 2.15 의 Keras model → Edge TPU compatible TFLite**: FaceNet 으로 검증
   됐지만, 우리 Keras 정의가 정확히 같은 op subset 출력하는지 별도 검증 필요.
3. **SCER 추가 후 학습 안정성**: Keras backbone 으로 SCER multi-loss 학습이
   PyTorch 와 동등하게 수렴할지 미지수.
4. **Coral USB Accelerator 보유**: 현재 수중에 있는지? 없으면 latency 비교 시
   "예상" vs "실측" 차이.

---

## 10. 결정 요약

> **Final Demo 의 TPU vs CPU 비교 요구로 모델 재구현 불가피.**
> Edge TPU 는 **toolchain (onnx_tf / onnx2tf → TF 2.15 → edgetpu_compiler v16)**
> level 에서 우리 PyTorch ResNet-18 ONNX 출력과 호환 안 됨. backbone 분리도
> 해결 못 함. **Keras-native ResNet-18 재구현 + PyTorch weight 이식 + SCER
> overlay** 가 필요. 5-7일 작업, v3 의 학습된 weights / corpus / aux_labels 모두 재활용.

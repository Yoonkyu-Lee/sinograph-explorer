# Phase 1 — Keras-native ResNet-18 reimplementation + PyTorch weight transfer

작성: 2026-04-27
참고: doc/25 (Edge TPU 차단 보고서), doc/24 §6 (SCER plan)

## 1. 목적

doc/25 의 결론 — onnx2tf / onnx_tf 어떤 경로로 변환해도 Edge TPU compiler v16
이 INT8 양자화 ops 를 인식하지 못함 — 을 우회하기 위해, 모델을 **Keras 로
직접 재구현** 하고 PyTorch best.pt 의 가중치만 옮긴다. Lab2 의 FaceNet 이
181/181 ops mapped 로 컴파일 성공한 동일 경로를 그대로 사용한다.

Phase 1 의 단일 검증 목표: **edgetpu_compiler 가 Keras-기반 ResNet-18 INT8
TFLite 를 100% (또는 ≥ 95%) ops mapped 로 컴파일** 한다.

성공 시 → Phase 2 (SCER 학습) 진입. 실패 시 → 모델/입력 사이즈 단순화 후 재시도.

## 2. 비목적 (out of scope)

- SCER 헤드 (radical / idc / strokes / 128-d embedding) 는 Phase 1 에서 다루지 않음.
- ArcFace 학습, soft structure filter 등 알고리즘 변경 없음.
- train_engine_v3/modules/model.py 와 train_loop.py 는 **건드리지 않음**.
- 정확도 평가는 Phase 1 범위 밖. parity (PyTorch ↔ Keras) 만 본다.

## 3. 영향 받는 파일

전부 신규 (수정 없음). PyTorch 측 코드는 그대로 둔다.

| 파일 | 용도 | 분량 |
|---|---|---|
| `train_engine_v3/modules/keras_resnet18.py` | Keras-native ResNet-18 (torchvision parity) | ~200 줄 |
| `train_engine_v3/scripts/40_port_pytorch_to_keras.py` | best.pt → keras 가중치 이식 | ~150 줄 |
| `train_engine_v3/scripts/41_export_keras_tflite.py` | Keras → SavedModel → INT8 TFLite | ~120 줄 |
| `train_engine_v3/scripts/42_verify_keras_parity.py` | PyTorch ↔ Keras forward 일치 검증 | ~80 줄 |

train_engine 자체는 손대지 않으므로 학습 코드와의 충돌 가능성은 없다.

## 4. 아키텍처 매핑 (torchvision → Keras)

torchvision `resnet18` 의 layer-by-layer 대응:

```
torchvision                                Keras 등가물
─────────────────────────────────────────  ──────────────────────────────────
Conv2d(3, 64, 7, stride=2, padding=3)      Conv2D(64, 7, strides=2, padding="same")
BatchNorm2d(64)                            BatchNormalization(epsilon=1e-5, momentum=0.9)
ReLU(inplace=True)                         ReLU()
MaxPool2d(3, stride=2, padding=1)          MaxPooling2D(3, strides=2, padding="same")

# layer1: 2× BasicBlock(64 → 64,  stride=1)
# layer2: 2× BasicBlock(64 → 128, stride=2 첫 블록)
# layer3: 2× BasicBlock(128→ 256, stride=2 첫 블록)
# layer4: 2× BasicBlock(256→ 512, stride=2 첫 블록)

BasicBlock:
  conv1 (3×3, optional stride=2) + bn1 + relu                      → Conv2D + BN + ReLU
  conv2 (3×3, stride=1)         + bn2                              → Conv2D + BN
  downsample (1×1 conv + bn) only if stride=2 or in_ch != out_ch   → optional Conv2D(1×1) + BN
  out = relu(conv2_out + identity)                                 → Add + ReLU

AdaptiveAvgPool2d(1)                       GlobalAveragePooling2D()
Flatten                                    Flatten()  (이미 (N,512) 면 생략)
Linear(512, num_classes)                   Dense(num_classes)
```

### 핵심 주의 사항

1. **Conv weight layout**: PyTorch (out_ch, in_ch, h, w) ↔ Keras (h, w, in_ch, out_ch).
   포팅 스크립트에서 `np.transpose(w, (2, 3, 1, 0))` 으로 변환.

2. **Padding**: torchvision 의 7×7 stride 2 + padding 3 은 Keras `padding="same"`
   과 입력 사이즈 128 에서 출력이 같다 (`128 / 2 = 64`). 3×3 stride 2 + padding 1
   도 동일. 단, **stride 1 + padding 1** 케이스도 padding="same" 으로 매핑된다.

3. **NCHW → NHWC**: PyTorch 입력 (N, 3, 128, 128), Keras/TF 입력 (N, 128, 128, 3).
   parity 검증 시 입력을 같은 normalization (`(x/255 - 0.5)/0.5`, [-1, 1]) 후
   PyTorch 는 NCHW, Keras 는 NHWC 로 넣는다.

4. **BatchNorm**: PyTorch default `momentum=0.1`, `eps=1e-5`. Keras 의 `momentum`
   은 EMA 계수의 의미가 반대 (Keras: `running = m·running + (1-m)·new`,
   PyTorch: `running = (1-m)·running + m·new`). 따라서 Keras 에서는
   `momentum=0.9` 를 쓰지만 — **inference (eval) 모드에서는 momentum 무관**,
   running_mean/running_var 만 사용. 우리는 학습이 끝난 가중치를 옮길 뿐이므로
   momentum 은 실제로 의미 없음.

5. **Bias**: torchvision Conv2d 는 BN 직전이라 `bias=False`. Keras 도 동일하게
   `use_bias=False`. 가중치 매핑이 단순해진다.

6. **Edge TPU 호환성**: FaceNet 이 컴파일된 사실로 보아, 표준 Keras 의
   Conv2D + BN + ReLU + Add + GlobalAveragePooling2D + Dense 조합은 Edge TPU
   에서 모두 native 매핑된다. 우리는 **이 사실에 베팅** 한다.

## 5. 가중치 이식 매핑

best.pt 의 state_dict 키 → keras_resnet18 의 layer name:

```
backbone.conv1.weight       → stem_conv.kernel
backbone.bn1.weight/bias    → stem_bn.gamma/beta
backbone.bn1.running_mean   → stem_bn.moving_mean
backbone.bn1.running_var    → stem_bn.moving_variance

backbone.layer1.0.conv1.weight       → layer1_block0_conv1.kernel
backbone.layer1.0.bn1.weight/bias    → layer1_block0_bn1.gamma/beta
backbone.layer1.0.conv2.weight       → layer1_block0_conv2.kernel
...

backbone.layer2.0.downsample.0.weight → layer2_block0_down_conv.kernel
backbone.layer2.0.downsample.1.*       → layer2_block0_down_bn.*

char_head.weight (n_class, 512)      → char_head.kernel  (transpose: (512, n_class))
char_head.bias                        → char_head.bias
```

aux 헤드 (radical/idc/strokes) 는 Phase 1 에서 무시. char-only inference 만 본다.

## 6. 실행 절차 (3 단계)

### Day 1 — 모델 빌드 + 가중치 이식 + parity 확인

1. `keras_resnet18.py` 작성 — KerasMultiHead 가 아닌 **CharOnly** 단일 출력 모델.
2. `40_port_pytorch_to_keras.py` 실행:
   - best.pt 로드 → state_dict 의 각 텐서를 Keras layer 에 set_weights.
   - Keras 모델을 `.keras` 파일로 저장 (`deploy_pi/export/v3_keras_char.keras`).
3. `42_verify_keras_parity.py` 실행:
   - 동일 calibration sample 100 장으로 PyTorch / Keras forward.
   - argmax top-1 일치율 ≥ 99%, max-abs-diff(logits) ≤ 1e-3 검증.
   - **불일치 시 BasicBlock / BN 매핑 재점검** 후 재시도.

### Day 2 — INT8 TFLite 변환 + Edge TPU 컴파일

4. `41_export_keras_tflite.py` 실행:
   - `.keras` → `tf.lite.TFLiteConverter.from_keras_model`
   - representative_dataset (300 샘플, NHWC, [-1, 1])
   - `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]`
   - `inference_input_type = inference_output_type = tf.int8`
   - 출력: `deploy_pi/export/v3_keras_char_int8.tflite` (~62 MB 추정)
5. `edgetpu_compiler v3_keras_char_int8.tflite -o deploy_pi/export/`
   - 결과 로그의 "Number of operations that will run on Edge TPU" 확인.
   - 목표: ≥ 95% (이상적으로 100%).

### Day 3 — Pi + Coral 측정 (선택, Phase 1 범위 밖)

성공 시 Pi 로 옮겨 `deploy_pi/scripts/infer_pi.py --tpu` (또는 동등) 로
TPU vs CPU latency 비교. 이는 Phase 4 의 데모이므로 일단 PC 단계에서 컴파일
성공을 확인하면 Phase 1 종료.

## 7. Decision tree

| Phase 1 결과 | 다음 단계 |
|---|---|
| ✅ ≥ 95% ops mapped, parity OK | Phase 2 (SCER 헤드 추가, train_engine 수정) 진입 |
| ⚠ 50-95% mapped | 어떤 op 이 unmapped 인지 확인 → block size 줄이거나 head 단순화 |
| ❌ < 50% mapped | input size 64 / 96 으로 축소 재시도, 그래도 안되면 **MobileNetV2** keras_application 으로 backbone 교체 |
| 💥 parity 실패 (PyTorch ≠ Keras) | BasicBlock downsample / BN momentum 매핑 재점검 |

## 8. 리스크

- **FC head (98169 classes × 512 → 50 MB)** 는 여전히 큰 텐서. FaceNet 처럼
  embedding (128-d) 만 출력하면 1.3 MB 라 무시할 수 있는데, char_head 는
  edgetpu compiler 의 silent fail 위험이 있다. 만약 char_head 만 unmapped
  되더라도 backbone + aux heads 가 TPU 위에 mapped 되면 충분하므로, Phase 2
  에서 SCER 로 가면 자연 해소될 가능성.
- TF 2.15 / Keras 의 BatchNormalization 이 INT8 변환 시 fold 되는지 확인 필요.
  일반적으로 fold 되지만 fold 가 안되면 unsupported op 가 추가됨.
- `padding="same"` 이 stride 2 일 때 PyTorch padding=1 과 결과가 어긋날 가능성
  (Keras 는 입력 우/아래쪽에 더 많이 패딩하는 경향). 사이즈 128 → 64 → 32
  순으로 짝수가 유지되므로 실제로는 동일해야 함. parity 단계에서 검출됨.

## 9. 산출물

- `train_engine_v3/modules/keras_resnet18.py`
- `train_engine_v3/scripts/40_port_pytorch_to_keras.py`
- `train_engine_v3/scripts/41_export_keras_tflite.py`
- `train_engine_v3/scripts/42_verify_keras_parity.py`
- `deploy_pi/export/v3_keras_char.keras`
- `deploy_pi/export/v3_keras_char_int8.tflite`
- `deploy_pi/export/v3_keras_char_int8_edgetpu.tflite` (목표)
- `doc/27_PHASE1_RESULTS.md` (Phase 1 종료 시 작성, edgetpu_compiler 로그 첨부)

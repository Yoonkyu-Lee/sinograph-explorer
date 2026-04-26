# Current Context from Lab 2

## Big Picture
Lab 2 covered the full edge-AI pipeline:
- train a model in PyTorch on the PC
- export to ONNX
- convert to TensorFlow SavedModel / TFLite
- quantize for deployment
- run inference on Raspberry Pi and Coral Edge TPU

The main lesson is that "working model" and "deployable model" are different milestones. Most of the real effort was in conversion, quantization, runtime compatibility, and hardware-specific debugging.

## Part 1 Takeaways: Raspberry Pi Environment Matters
- Raspberry Pi work was easiest when treated as a separate deployment target, not just an extension of the PC environment.
- On Pi, `pyenv` environment management mattered a lot. The lab used an `ECE479` Python 3.10 environment.
- For Edge TPU work, the Pi runtime typically used `tflite_runtime`, not full TensorFlow.
- Camera support was fragile across environments. A camera could work with `rpicam-still` while failing with OpenCV `VideoCapture`.

## Part 2 Takeaways: Training vs Deployment
- The baseline classifier was a small Fashion-MNIST CNN trained in PyTorch.
- Model artifacts were standardized into a regular naming scheme:
  - `model_1x1.*`
  - `model_2x2.*`
  - `model_4x4.*`
- The useful export path was:
  - `.pth` -> `.onnx` -> TensorFlow SavedModel -> `.tflite`
- Two quantization concepts were important:
  - dynamic range quantization: smaller/easier, mainly CPU-oriented
  - full integer quantization: required for Edge TPU compilation

## Part 2 Performance Lessons
- Edge TPU is not automatically faster for tiny workloads.
- For `1x1`, TPU could be slower than CPU because transfer/invocation overhead dominates.
- Spatial batching (`2x2`, `4x4`) made TPU speedups meaningful by amortizing overhead across multiple images.
- Therefore, throughput and latency depend not only on the model, but also on input packing strategy.

## Part 2 Reporting Lessons
- Benchmark scripts that use random tensors are valid for latency measurement, but not for accuracy claims.
- Rubric coverage requires more than a speed table:
  - train/test accuracy
  - loss/accuracy plots
  - quantization comparison
  - CPU vs TPU timing
  - interpretation of results

## Part 3 Takeaways: Reconstructing and Deploying a Face Model
- The Inception-ResNet-v1 model had to be reconstructed from a reference ONNX graph plus pretrained weights.
- Important architecture pieces:
  - `BasicConv2d`
  - stem
  - `InceptionResnetA/B/C`
  - reduction blocks
  - full repeated-block assembly
- A successful weight load was the key correctness check for the reconstructed model.

## Part 3 Conversion Lessons
- ONNX export can succeed even when the requested lower opset conversion fails; the actual saved model may remain at a newer opset.
- `onnx2tf` solved layout conversion issues between PyTorch/ONNX and TensorFlow/TFLite.
- Converting to deployable TFLite often involved more dependency debugging than model debugging.
- Final deployable face model artifact:
  - `facenet_fullint.tflite`

## Part 3 Verification Lessons
- Face verification pipeline components:
  - face detection
  - loose crop with margin
  - square resize to `160x160`
  - normalization to `[-1, 1]`
  - int8 quantization / dequantization for TFLite
  - embedding comparison with Euclidean distance and cosine similarity
- Same-identity image pairs should rank as top matches if the pipeline is working correctly.

## Deployment Lessons from Pi
- Splitting responsibilities across environments was often the cleanest solution.
- Practical split:
  - system/Pi camera tools for capture
  - `ECE479` environment for inference
- Edge TPU delegate loading worked once the correct runtime was available.
- Camera preview and model inference do not necessarily belong in the same Python environment on Pi.

## File / Workflow Habits Worth Reusing in Lab 3
- Keep naming conventions consistent early.
- Prefer relative paths inside notebooks and scripts.
- Validate each stage independently:
  - model build
  - weight load
  - conversion
  - quantization
  - inference
  - hardware acceleration
- When debugging, separate:
  - model errors
  - environment/package errors
  - hardware/runtime errors
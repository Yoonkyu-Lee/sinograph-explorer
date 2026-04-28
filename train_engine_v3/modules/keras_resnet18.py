"""Keras-native ResNet-18 + char head, layer-by-layer parity with torchvision.

Why Keras-native: doc/25 documented that onnx2tf / onnx_tf converted INT8
TFLite is rejected by edgetpu_compiler v16 ("unsupported data type" on every
op). Lab2 FaceNet (Keras → SavedModel → INT8) compiles 181/181 ops mapped,
so we follow the same path.

Architecture (matches torchvision.models.resnet18 exactly):

    input (N, H, W, 3)              # NHWC
        ↓ Conv2D(64, 7×7, stride=2, padding="same") + BN + ReLU
        ↓ MaxPool2D(3×3, stride=2, padding="same")
    layer1: 2× BasicBlock(64,  stride=1)            # no downsample (in == out)
    layer2: 2× BasicBlock(128, stride=2 first)      # downsample at block 0
    layer3: 2× BasicBlock(256, stride=2 first)
    layer4: 2× BasicBlock(512, stride=2 first)
        ↓ GlobalAveragePooling2D                    # (N, 512)
        ↓ Dense(num_classes)                        # char_head
    logits (N, num_classes)

Only the char head is included (no aux heads). Phase 1 verifies Edge TPU
compilation path; SCER heads are Phase 2.

Layer naming follows the PyTorch state_dict structure so the weight porter
can do a string-based mapping:

    backbone.conv1.weight       ↔ stem_conv
    backbone.bn1.*              ↔ stem_bn
    backbone.layerX.Y.convN.*   ↔ layerX_blockY_convN
    backbone.layerX.Y.bnN.*     ↔ layerX_blockY_bnN
    backbone.layerX.Y.downsample.0.* ↔ layerX_blockY_down_conv
    backbone.layerX.Y.downsample.1.* ↔ layerX_blockY_down_bn
    char_head.weight/bias       ↔ char_head
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

# torchvision BasicBlock has expansion=1; resnet18 stage configs are 4 stages
# of 2 blocks each, channels = [64, 128, 256, 512], first block of stages 2-4
# does the spatial downsample (stride=2).
STAGES = [
    # (out_channels, num_blocks, stride_of_first_block)
    (64,  2, 1),
    (128, 2, 2),
    (256, 2, 2),
    (512, 2, 2),
]


def _basic_block(x, out_ch: int, stride: int, name: str):
    """torchvision BasicBlock: conv-bn-relu-conv-bn + identity → relu.

    Padding parity note:
      torchvision Conv2d(..., padding=1) pads symmetrically (1 on each side).
      Keras "same" pads ASYMMETRICALLY when stride>1 + even input (0 left,
      1 right), shifting the receptive field by 1 pixel. To match torchvision
      we use ZeroPadding2D(1) + padding="valid" for the stride=2 conv1.
      For stride=1, "same" with kernel 3 produces the same symmetric (1,1) pad.
    """
    in_ch = x.shape[-1]
    identity = x

    if stride == 1:
        out = layers.Conv2D(
            out_ch, 3, strides=1, padding="same", use_bias=False,
            name=f"{name}_conv1",
        )(x)
    else:
        # stride=2 + 3x3 + sym pad=1: explicit pad to match torchvision
        padded = layers.ZeroPadding2D(padding=1, name=f"{name}_pad1")(x)
        out = layers.Conv2D(
            out_ch, 3, strides=stride, padding="valid", use_bias=False,
            name=f"{name}_conv1",
        )(padded)
    out = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{name}_bn1")(out)
    out = layers.ReLU(name=f"{name}_relu1")(out)

    out = layers.Conv2D(
        out_ch, 3, strides=1, padding="same", use_bias=False,
        name=f"{name}_conv2",
    )(out)
    out = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{name}_bn2")(out)

    if stride != 1 or in_ch != out_ch:
        # downsample is 1x1 conv (no padding); valid == same here, but keep
        # valid to be explicit. stride=2 + 1x1 picks every other pixel
        # starting at 0 — same as torchvision.
        identity = layers.Conv2D(
            out_ch, 1, strides=stride, padding="valid", use_bias=False,
            name=f"{name}_down_conv",
        )(identity)
        identity = layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9, name=f"{name}_down_bn",
        )(identity)

    out = layers.Add(name=f"{name}_add")([out, identity])
    out = layers.ReLU(name=f"{name}_relu2")(out)
    return out


def build_keras_resnet18_char(
    num_classes: int,
    input_size: int = 128,
) -> Model:
    """Build the Keras char-only ResNet-18.

    Input  : (N, input_size, input_size, 3) float32 (caller pre-normalizes to [-1, 1])
    Output : (N, num_classes) float32 logits
    """
    inp = layers.Input(shape=(input_size, input_size, 3), name="input")

    # Stem: 7×7 conv stride 2 (sym pad=3), BN, ReLU, 3×3 maxpool stride 2 (sym pad=1)
    # See _basic_block padding-parity note: Keras "same" + stride 2 picks
    # different windows than torchvision symmetric padding, so we use explicit
    # ZeroPadding2D + valid to guarantee parity.
    x = layers.ZeroPadding2D(padding=3, name="stem_pad")(inp)
    x = layers.Conv2D(
        64, 7, strides=2, padding="valid", use_bias=False, name="stem_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="valid", name="stem_pool")(x)

    # Stages
    for stage_idx, (out_ch, n_blocks, first_stride) in enumerate(STAGES, start=1):
        for block_idx in range(n_blocks):
            stride = first_stride if block_idx == 0 else 1
            x = _basic_block(
                x, out_ch=out_ch, stride=stride,
                name=f"layer{stage_idx}_block{block_idx}",
            )

    # GAP + Dense (char head)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    logits = layers.Dense(num_classes, name="char_head")(x)

    return Model(inputs=inp, outputs=logits, name="keras_resnet18_char")


def block_layer_names(stage_idx: int, block_idx: int, has_downsample: bool) -> dict:
    """Return the {pytorch_key_prefix: keras_layer_name} mapping for one block."""
    base = f"layer{stage_idx}_block{block_idx}"
    pt_base = f"backbone.layer{stage_idx}.{block_idx}"
    out = {
        f"{pt_base}.conv1": f"{base}_conv1",
        f"{pt_base}.bn1":   f"{base}_bn1",
        f"{pt_base}.conv2": f"{base}_conv2",
        f"{pt_base}.bn2":   f"{base}_bn2",
    }
    if has_downsample:
        out[f"{pt_base}.downsample.0"] = f"{base}_down_conv"
        out[f"{pt_base}.downsample.1"] = f"{base}_down_bn"
    return out


def all_layer_name_mapping() -> dict:
    """Full mapping {pytorch_key_prefix: keras_layer_name} for the entire net."""
    m = {
        "backbone.conv1": "stem_conv",
        "backbone.bn1":   "stem_bn",
        "char_head":      "char_head",
    }
    for stage_idx, (out_ch, n_blocks, first_stride) in enumerate(STAGES, start=1):
        for block_idx in range(n_blocks):
            has_down = (block_idx == 0 and (first_stride != 1 or stage_idx > 1))
            # stage1 stride=1 + in==out → no downsample
            if stage_idx == 1:
                has_down = False
            m.update(block_layer_names(stage_idx, block_idx, has_down))
    return m

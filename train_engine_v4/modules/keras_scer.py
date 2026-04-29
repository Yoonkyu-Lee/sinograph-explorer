"""Keras-native SCER (deploy variant) — Phase 3 (doc/29 §7).

Mirrors `train_engine_v3/modules/keras_resnet18.py` but for the SCER
deploy graph: backbone + 4 structure heads + 128-d L2-normalized embedding.
char_head and arc_classifier are explicitly NOT in this model
(they live only in the training-time PyTorch graph).

Output dict (in fixed order, since Keras model outputs are positional):

    [0] embedding         (N, 128)  — L2-normalized
    [1] radical           (N, 214)
    [2] total_strokes     (N,)      — scalar regression
    [3] residual_strokes  (N,)      — scalar regression
    [4] idc               (N, 12)

Padding parity (doc/27 §1.2): torchvision Conv2D pads symmetrically;
Keras "same" + stride>1 is asymmetric. We use ZeroPadding2D + valid for
all stride>1 + 3×3/7×7 layers to preserve receptive-field offsets.

L2 normalize: `tf.math.l2_normalize` lowers to (x / sqrt(sum(x²)+eps)) →
edgetpu_compiler v16 should handle it. Fallback: drop the L2 op and do
normalization on the Pi CPU side before cosine search (1 division per query).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

NUM_RADICALS = 214
NUM_IDC = 12
EMB_DIM = 128

# torchvision resnet18 stage configs
STAGES = [
    # (out_channels, num_blocks, stride_of_first_block)
    (64,  2, 1),
    (128, 2, 2),
    (256, 2, 2),
    (512, 2, 2),
]


def _basic_block(x, out_ch: int, stride: int, name: str):
    """torchvision BasicBlock — sym-pad ZeroPadding2D + valid for stride>1."""
    in_ch = x.shape[-1]
    identity = x

    if stride == 1:
        out = layers.Conv2D(
            out_ch, 3, strides=1, padding="same", use_bias=False,
            name=f"{name}_conv1",
        )(x)
    else:
        padded = layers.ZeroPadding2D(padding=1, name=f"{name}_pad1")(x)
        out = layers.Conv2D(
            out_ch, 3, strides=stride, padding="valid", use_bias=False,
            name=f"{name}_conv1",
        )(padded)
    out = layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                      name=f"{name}_bn1")(out)
    out = layers.ReLU(name=f"{name}_relu1")(out)

    out = layers.Conv2D(
        out_ch, 3, strides=1, padding="same", use_bias=False,
        name=f"{name}_conv2",
    )(out)
    out = layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                      name=f"{name}_bn2")(out)

    if stride != 1 or in_ch != out_ch:
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


def build_keras_scer(
    num_radicals: int = NUM_RADICALS,
    num_idc: int = NUM_IDC,
    emb_dim: int = EMB_DIM,
    input_size: int = 128,
    l2_normalize_in_graph: bool = True,
    batch_size: int | None = None,
) -> Model:
    """Build the Keras SCER deploy model.

    Args:
        num_radicals, num_idc, emb_dim: dimensions
        input_size: spatial input size (default 128)
        l2_normalize_in_graph: if True, apply tf.math.l2_normalize to the
            embedding output INSIDE the graph. If False, return raw
            (un-normalized) embedding — caller (Pi CPU) does L2 norm.
            Set to False if Edge TPU compilation rejects the L2-norm op.
        batch_size: if given (e.g. 1), the input tensor has FIXED batch dim,
            which is required by edgetpu_compiler v16 (rejects dynamic-sized
            tensors). For training/parity testing pass None to use Keras
            default (dynamic batch). For deploy export pass 1.

    Input  : (B, input_size, input_size, 3) float32, expected in [-1, 1]
    Outputs: 5 tensors:
        [0] embedding         (B, 128)  L2-normalized
        [1] radical           (B, 214)
        [2] total_strokes     (B, 1)    — kept as (B,1) NOT squeezed
        [3] residual_strokes  (B, 1)
        [4] idc               (B, 12)
    """
    if batch_size is not None:
        inp = layers.Input(shape=(input_size, input_size, 3),
                            batch_size=batch_size, name="input")
    else:
        inp = layers.Input(shape=(input_size, input_size, 3), name="input")

    # ---- Stem (parity with torchvision) ----
    x = layers.ZeroPadding2D(padding=3, name="stem_pad")(inp)
    x = layers.Conv2D(
        64, 7, strides=2, padding="valid", use_bias=False, name="stem_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="valid", name="stem_pool")(x)

    # ---- Backbone stages ----
    for stage_idx, (out_ch, n_blocks, first_stride) in enumerate(STAGES, start=1):
        for block_idx in range(n_blocks):
            stride = first_stride if block_idx == 0 else 1
            x = _basic_block(
                x, out_ch=out_ch, stride=stride,
                name=f"layer{stage_idx}_block{block_idx}",
            )

    # ---- GAP → feature ----
    feat = layers.GlobalAveragePooling2D(name="gap")(x)              # (N, 512)

    # ---- 4 structure heads (mirroring torchvision linear) ----
    # Note: total/residual stroke heads keep their (B, 1) shape (NOT squeezed
    # to (B,) — the Reshape((),) op makes a fully-dynamic tensor that
    # edgetpu_compiler rejects). Caller squeezes after dequantize on CPU.
    radical = layers.Dense(num_radicals, name="radical_head")(feat)
    total_strokes = layers.Dense(1, name="total_strokes_head")(feat)
    residual = layers.Dense(1, name="residual_head")(feat)
    idc = layers.Dense(num_idc, name="idc_head")(feat)

    # ---- embedding head ----
    emb_raw = layers.Dense(emb_dim, name="embedding_head")(feat)
    if l2_normalize_in_graph:
        # tf.math.l2_normalize(x, axis=-1) = x / sqrt(sum(x^2)+eps)
        # Wrapped in a Lambda for clean naming + isolation in graph dump.
        embedding = layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=-1, epsilon=1e-12),
            name="embedding_l2",
        )(emb_raw)
    else:
        embedding = layers.Activation("linear", name="embedding_raw")(emb_raw)

    return Model(
        inputs=inp,
        outputs=[embedding, radical, total_strokes, residual, idc],
        name="keras_scer",
    )


def block_layer_names(stage_idx: int, block_idx: int, has_downsample: bool) -> dict:
    """Return {pytorch_key_prefix: keras_layer_name} for one block."""
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
    """Full {pytorch_key_prefix: keras_layer_name} for the SCER deploy graph.

    Excludes char_head and arc_classifier (NOT in deploy).
    """
    m = {
        "backbone.conv1":           "stem_conv",
        "backbone.bn1":             "stem_bn",
        "radical_head":             "radical_head",
        "total_strokes_head":       "total_strokes_head",
        "residual_head":            "residual_head",
        "idc_head":                 "idc_head",
        "embedding_head":           "embedding_head",
    }
    for stage_idx, (_out_ch, n_blocks, first_stride) in enumerate(STAGES, start=1):
        for block_idx in range(n_blocks):
            has_down = (block_idx == 0 and (first_stride != 1 or stage_idx > 1))
            if stage_idx == 1:
                has_down = False
            m.update(block_layer_names(stage_idx, block_idx, has_down))
    return m

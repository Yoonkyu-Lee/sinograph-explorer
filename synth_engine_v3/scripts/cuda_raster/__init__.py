"""GPU polygon rasterizer for v3 mask generation.

Phase OPT-2 (doc/20). Replaces PIL/freetype CPU rasterization (~1.4 ms /
glyph wall time, mask_wait bottleneck) with batched on-device rasterization.

Public API:
    outline_cache.get_outline(font_path, face_index, char) → OutlineData
    rasterize.rasterize_batch(outlines, H, W) → torch.Tensor (B, 1, H, W)
"""

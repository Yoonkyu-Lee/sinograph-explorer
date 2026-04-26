# Legacy prototype scripts

These scripts are **superseded by `generate.py`**. Kept here only as reference
for how v1 was initially prototyped as two parallel paths.

- `render_systemfonts.py` — replaced by `generate.py <char> --effects clean`
- `render_stylized.py`    — replaced by `generate.py <char> --sources font:malgun --effects all`

The conceptual change: "fonts" and "stylized" are no longer separate pipelines.
Instead `generate.py` has one pipeline with two pluggable axes
(base source × effect stack), and the former behaviors are special cases of
that matrix.

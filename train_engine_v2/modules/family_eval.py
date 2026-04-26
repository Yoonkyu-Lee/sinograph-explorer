"""Family-aware accuracy using sinograph_canonical_v2.sqlite.

Partial-credit metric: a prediction is
  - exact (1.0)    if pred_codepoint == target_codepoint
  - family (0.5)   if pred_codepoint is in target's variant family (family_members_json)
  - miss (0.0)     otherwise

Uses `variant_components.family_members_json` which is the post-fix family
(detail-only relations like synonyms/opposites/alt_forms already excluded,
see sinograph_canonical_v2 build script).

`canonical_family_members_json` is stricter (corroborated-only);
`enriched_family_members_json` is wider. We default to the middle one.
"""
from __future__ import annotations

import json
import sqlite3

import torch


def load_family_map(db_path, field="family_members_json"):
    """Return dict: codepoint_notation -> set of family codepoint notations.

    Only codepoints present in variant_components are keys. Callers must
    handle missing keys (fallback = singleton).
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    fam = {}
    q = f"SELECT codepoint, {field} FROM variant_components"
    for cp, j in cur.execute(q):
        members = json.loads(j) if j else [cp]
        fam[cp] = set(members)
    con.close()
    return fam


@torch.no_grad()
def family_aware_accuracy(model, loader, device, class_index, canonical_db_path,
                          family_field="family_members_json"):
    """Compute exact / family / weighted accuracy.

    Args:
      class_index: dict notation -> class_idx (same as saved by 20_train.py)
    """
    idx_to_notation = {i: n for n, i in class_index.items()}
    fam_map = load_family_map(canonical_db_path, field=family_field)

    model.eval()
    n = 0
    exact = 0
    family = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_cpu = y.cpu().tolist()
        for p, t in zip(pred, y_cpu):
            n += 1
            if p == t:
                exact += 1
                continue
            t_not = idx_to_notation[t]
            p_not = idx_to_notation[p]
            members = fam_map.get(t_not, {t_not})
            if p_not in members:
                family += 1

    n = max(n, 1)
    return {
        "exact": exact / n,
        "family_only": family / n,
        "weighted": (exact + 0.5 * family) / n,
        "n": n,
    }

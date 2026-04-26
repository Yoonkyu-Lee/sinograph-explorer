"""Single-image inference for a trained ckpt.

Usage:
  python 30_predict.py --ckpt .../ckpt_epoch_05.pth \
                       --class-index .../class_index.json \
                       --config .../resnet18_t1_mini.yaml \
                       --image path/to/photo.png \
                       [--topk 5] [--family-db .../canonical_v2.sqlite]

Output: top-k codepoint + literal char + confidence + (optional) variant family.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Windows terminal often defaults to cp1252 which can't encode CJK chars.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--config", required=True,
                    help="training config yaml (used for model/input_size)")
    ap.add_argument("--image", required=True, help="path to input image file")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--family-db", default=None,
                    help="path to canonical_v2.sqlite for variant family lookup")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    class_index = json.load(open(args.class_index, encoding="utf-8"))
    # class_index is {notation: idx}; reverse to idx-ordered list of notations
    classes = sorted(class_index.keys(), key=lambda k: class_index[k])
    num_classes = len(classes)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"[predict] device={device} classes={num_classes} ckpt={args.ckpt}")

    model = build_model(cfg["model"]["name"], num_classes).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    # Preprocessing — must match the val pipeline used during training.
    # Training pipeline (M4 winner): PIL → Resize → CenterCrop → PILToTensor (uint8)
    # then GPU side: float/255 → (x - 0.5) / 0.5
    input_size = cfg["model"]["input_size"]
    cpu_t = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.PILToTensor(),  # uint8 [3, H, W]
    ])
    img_pil = Image.open(args.image).convert("RGB")
    print(f"[predict] image: {args.image}  raw size: {img_pil.size}")
    x = cpu_t(img_pil).unsqueeze(0).to(device)        # [1, 3, H, W] uint8
    x = x.float().div_(255.0).sub_(0.5).div_(0.5)      # GPU normalize

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs[0].topk(args.topk)

    print(f"\n=== Top-{args.topk} predictions ===")
    top_notations = []
    for rank, (p, i) in enumerate(zip(top_p.cpu().tolist(), top_i.cpu().tolist())):
        notation = classes[i]
        try:
            ch = chr(int(notation[2:], 16))
        except Exception:
            ch = "?"
        top_notations.append(notation)
        print(f"  #{rank + 1}  {notation}  '{ch}'  prob={p * 100:.1f}%")

    # optional: variant family of top-1
    if args.family_db:
        con = sqlite3.connect(args.family_db)
        row = con.execute(
            "SELECT family_members_json FROM variant_components WHERE codepoint=?",
            (top_notations[0],),
        ).fetchone()
        con.close()
        if row and row[0]:
            members = json.loads(row[0])
            chars = []
            for m in members:
                try:
                    chars.append(chr(int(m[2:], 16)))
                except Exception:
                    chars.append("?")
            print(f"\n[family] top-1 {top_notations[0]} variant family ({len(members)}): "
                  f"{' '.join(chars)}")
        else:
            print(f"\n[family] top-1 {top_notations[0]} not found in canonical DB")


if __name__ == "__main__":
    main()

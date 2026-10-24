"""Track 1 inference entry point of team tinyreef-praneel: TinyReefNet.

The organizers call   main(model_dir, input_dir, output_dir, device)   and nothing else.
Same pipeline as tinyreefnet/infer.py (which produced our leaderboard submission): resize to the network
resolution (bilinear), one forward pass, restored image rounded to uint8, argmax mask, both resized to the
scoring size (longest side at most 1024 px, downscale only).
"""
import os

import numpy as np
import torch
from PIL import Image

from src.tinyreef import build_model
from src.tinyreef.utils import scoring_size

CKPT_NAME = "tinyreef_best.pth"


def _find_checkpoint(model_dir):
    path = os.path.join(model_dir, CKPT_NAME)
    if os.path.isfile(path):
        return path
    # a downloaded weights file may have been saved under another name: accept the only file that is not documentation
    found = [f for f in sorted(os.listdir(model_dir)) if f not in ("weights_link.yaml", "README.md", ".gitkeep")]
    if len(found) != 1:
        raise FileNotFoundError(f"expected {CKPT_NAME} (or exactly one weights file) in {model_dir}, found {found}")
    return os.path.join(model_dir, found[0])


@torch.no_grad()
def main(model_dir, input_dir, output_dir, device):
    ck = torch.load(_find_checkpoint(model_dir), map_location=device, weights_only=True)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    H, W = cfg["data"]["resolution"]

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        with Image.open(os.path.join(input_dir, name)) as im:
            im = im.convert("RGB")
            tw, th = scoring_size(*im.size)
            small = im.resize((W, H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(small, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        restored, logits = model(x)
        r = (restored[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        Image.fromarray(r).resize((tw, th), Image.BILINEAR).save(os.path.join(output_dir, "restored", stem + ".png"))
        m = logits[0].argmax(0).byte().cpu().numpy()
        Image.fromarray(m).resize((tw, th), Image.NEAREST).save(os.path.join(output_dir, "seg_masks", stem + ".png"))

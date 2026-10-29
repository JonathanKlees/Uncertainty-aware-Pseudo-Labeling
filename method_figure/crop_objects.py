#!/usr/bin/env python3
"""
Crop the marked zebra/giraffe boxes from abstract_figure.ipynb out of the
original COCO images, so they can be reused as graph nodes in method_figure.py.

Box coordinates below are copied directly from abstract_figure/abstract_figure.ipynb
(composite-figure coordinate space); image placements ("images" list / extent)
are copied from the same notebook cell.
"""

import os
import cv2
import numpy as np
from PIL import Image

COCO_ROOT = "/home/klees/data/coco/val2017/"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# (image_path, extent=(x, y, w, h) as used in ax.imshow(..., extent=(x,x+w,y,y+h)))
IMAGES = {
    "img1": (COCO_ROOT + "000000064359.jpg", (3.25, 0.5, 3, 2)),   # giraffes & zebras
    "img2": (COCO_ROOT + "000000052462.jpg", (0, -0.5, 3, 2)),      # two zebras
    "img3": (COCO_ROOT + "000000157098.jpg", (6.5, -0.5, 3, 2)),    # three giraffes
}

# img1 was shot in heavy shade / backlight and comes out noticeably darker
# and flatter than img2/img3 — every crop taken from it gets a lighting
# fix (see enhance_lighting) so all nodes read consistently in the figure.
DARK_IMAGE_KEYS = {"img1"}


def enhance_lighting(im, gamma=0.75, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Brighten and recover local contrast in an underexposed crop.

    CLAHE (on the LAB lightness channel) pulls detail out of the shadows
    without blowing out the already-bright areas; the gamma lift on top
    corrects the overall underexposure. Gamma < 1 brightens.
    """
    rgb = np.array(im.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    rgb = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    lut = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    rgb = cv2.LUT(rgb, lut)

    return Image.fromarray(rgb)

# box_name: (image_key, (x, y, w, h) in composite coords, class)
BOXES = {
    "zebra_gt_1":     ("img1", (4.515, 1.05, 0.8, 0.8),  "zebra"),
    "zebra_prop_1":   ("img2", (0.68, 0.0, 0.68, 0.95),  "zebra"),
    "zebra_prop_2":   ("img2", (1.87, 0.05, 0.6, 1.0),   "zebra"),
    "zebra_prop_4":   ("img1", (5.5, 1.0, 0.95, 0.58),   "zebra"),

    "giraffe_gt_1":   ("img3", (7.1, -0.49, 0.65, 1.9),  "giraffe"),
    "giraffe_prop_1": ("img3", (7.7, -0.49, 1.03, 1.9),  "giraffe"),
    "giraffe_prop_4": ("img1", (4.9, 1.7, 0.75, 0.7),    "giraffe"),
    "giraffe_prop_5": ("img1", (4.4, 1.8, 0.7, 0.69),    "giraffe"),
}

# false-positive "background clutter" crops used in method_figure.py — just
# tree/foliage patches sub-cropped from giraffe_prop_1's own background.
# They're deep-shadow foliage so they read too dark to make out; brighten
# them the same way as the dark-image crops below.
FP_TREE_BOXES = {
    "fp_tree_1": ("giraffe_prop_1", (110, 0, 192, 80)),
    "fp_tree_2": ("giraffe_prop_1", (110, 0, 192, 55)),
}


def crop_box(image_path, extent, box, pad_frac=0.06):
    img_x, img_y, img_w, img_h = extent
    box_x, box_y, box_w, box_h = box

    im = Image.open(image_path).convert("RGB")
    W, H = im.size

    left_frac = (box_x - img_x) / img_w
    right_frac = (box_x + box_w - img_x) / img_w
    bottom_frac = (box_y - img_y) / img_h
    top_frac = (box_y + box_h - img_y) / img_h

    px_left = left_frac * W
    px_right = right_frac * W
    px_top = (1 - top_frac) * H
    px_bottom = (1 - bottom_frac) * H

    pad_x = (px_right - px_left) * pad_frac
    pad_y = (px_bottom - px_top) * pad_frac
    px_left -= pad_x
    px_right += pad_x
    px_top -= pad_y
    px_bottom += pad_y

    px_left = max(0, px_left)
    px_top = max(0, px_top)
    px_right = min(W, px_right)
    px_bottom = min(H, px_bottom)

    return im.crop((int(px_left), int(px_top), int(px_right), int(px_bottom)))


if __name__ == "__main__":
    crops = {}
    for name, (img_key, box, cls) in BOXES.items():
        img_path, extent = IMAGES[img_key]
        crop = crop_box(img_path, extent, box)
        if img_key in DARK_IMAGE_KEYS:
            crop = enhance_lighting(crop)
        crops[name] = crop
        out_path = os.path.join(OUT_DIR, f"{name}.png")
        crop.save(out_path)
        print(f"Saved {out_path}  size={crop.size}  class={cls}")

    for name, (src_name, px_box) in FP_TREE_BOXES.items():
        crop = enhance_lighting(crops[src_name].crop(px_box))
        out_path = os.path.join(OUT_DIR, f"{name}.png")
        crop.save(out_path)
        print(f"Saved {out_path}  size={crop.size}  class=fp")

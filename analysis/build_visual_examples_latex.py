"""Generates LaTeX \\figure blocks for visual_soft_label_examples.ipynb's example
figures (agreement / disagreement / uncertain ground truth), one full-width figure*
per category, 2 subfigures per row, via the subcaption package. One self-contained
.tex snippet per category, saved to tables/, so each can be \\input{} independently.

Also generates the appendix page for the other datasets (visual_soft_label_examples.py
--dataset ...): one full-width figure* per category, 2 rows x 3 columns, one column
per dataset (Pascal VOC | Cityscapes | KITTI) with two hand-picked examples each,
saved together to tables/visual_examples_other_datasets.tex.
"""
import os

CATEGORIES = {
    "agreement": dict(
        n=4, caption=r"Examples where our method and Grounding DINO agree on the predicted class.",
        label="fig:visual-agreement",
    ),
    "disagreement": dict(
        n=4, caption=r"Examples where our method and Grounding DINO disagree on the predicted class.",
        label="fig:visual-disagreement",
    ),
    "uncertain_gt": dict(
        n=4, caption=r"Cases where the soft ground truth itself is not fully confident and "
                      r"neither are our method and Grounding DINO.",
        label="fig:visual-uncertain-gt",
    ),
}

SUBFIG_WIDTH = 0.29   # 2 per row in a full-width figure*; largest width at which all three
                       # COCO figures fit on one A4 float page (IEEEtran) with their captions
IMG_DIR = "imgs/examples"   # paper repo's image directory -- not this repo's plots/examples

# other datasets: per category, the two example indices shown per dataset (hand-picked
# from the 4 generated per category for legibility and variety); a (category, index) pair
# takes an image generated for another category instead. Disagreement shows one "GDino was
# right" case for Cityscapes only (Pascal VOC's is a tiny, ambiguous box); for KITTI it uses
# a partially visible vehicle from the uncertain-GT set, on which our estimate and GDino
# also disagree.
OTHER_DATASETS = {"voc_trainval": "Pascal VOC", "cityscapes_trainval": "Cityscapes", "kitti_train": "KITTI"}
OTHER_SELECTION = {
    "agreement": {"voc_trainval": [0, 1], "cityscapes_trainval": [0, 3], "kitti_train": [1, 3]},
    "disagreement": {"voc_trainval": [0, 2], "cityscapes_trainval": [2, 3], "kitti_train": [2, ("uncertain_gt", 0)]},
    "uncertain_gt": {"voc_trainval": [1, 2], "cityscapes_trainval": [0, 2], "kitti_train": [2, 3]},
}
OTHER_CAPTIONS = {
    "agreement": r"Agreement examples on Pascal VOC, Cityscapes and KITTI: cases where our "
                 r"method's estimate and GDino's calibrated confidence agree on the predicted class.",
    "disagreement": r"Disagreement examples on Pascal VOC, Cityscapes and KITTI: cases where our "
                    r"method's estimate and GDino's calibrated confidence disagree on the predicted "
                    r"class. In the bottom Cityscapes example, GDino is correct and our estimate is not.",
    "uncertain_gt": r"Uncertain ground truth examples on Pascal VOC, Cityscapes and KITTI: cases "
                    r"where the soft ground truth itself is not fully confident (max class "
                    r"probability $< 0.9$).",
}
OTHER_SUBFIG_WIDTH = 0.28   # 3 per row on the full text width (figure*). All three figures
                            # must fit on one float page together (IEEEtran, A4): 0.29 is the
                            # largest that does, 0.28 leaves ~one caption line of slack.


def build_figure(name, cfg):
    lines = [r"\begin{figure*}[p]", r"    \centering", r"    \captionsetup[subfigure]{skip=1pt}"]
    for i in range(cfg["n"]):
        lines.append(rf"    \begin{{subfigure}}[b]{{{SUBFIG_WIDTH}\textwidth}}")
        lines.append(rf"        \includegraphics[width=\linewidth]{{{IMG_DIR}/{name}_{i}.png}}")
        lines.append(r"        \caption{}")
        lines.append(rf"        \label{{fig:visual-{name.replace('_', '-')}-{i}}}")
        lines.append(r"    \end{subfigure}")
        if i % 2 == 0:
            lines.append(r"    \hspace{2em}")
        elif i != cfg["n"] - 1:
            lines.append(r"    \par")
    lines += [
        rf"    \caption{{{cfg['caption']}}}",
        rf"    \label{{{cfg['label']}}}",
        r"\end{figure*}",
    ]
    return "\n".join(lines)


def other_image(name, key, row):
    """(category, index) of the image shown in figure `name` for dataset `key`, row `row`."""
    sel = OTHER_SELECTION[name][key][row]
    return sel if isinstance(sel, tuple) else (name, sel)


def build_other_datasets_figure(name):
    """2 x 3 grid, row r holds each dataset's r-th selected example, so every column is
    one dataset; the dataset name goes in each subcaption."""
    lines = [r"\begin{figure*}[p]", r"    \centering", r"    \captionsetup[subfigure]{skip=1pt}"]
    for row in range(2):
        for col, (key, display_name) in enumerate(OTHER_DATASETS.items()):
            category, i = other_image(name, key, row)
            lines.append(rf"    \begin{{subfigure}}[b]{{{OTHER_SUBFIG_WIDTH}\textwidth}}")
            lines.append(rf"        \includegraphics[width=\linewidth]{{{IMG_DIR}/{key}/{category}_{i}.png}}")
            lines.append(rf"        \caption{{{display_name}}}")
            lines.append(rf"        \label{{fig:visual-{name.replace('_', '-')}-{key.split('_')[0]}-{row}}}")
            lines.append(r"    \end{subfigure}")
            if col < len(OTHER_DATASETS) - 1:
                lines.append(r"    \hfill")
        if row == 0:
            lines.append(r"    \par")
    lines += [
        rf"    \caption{{{OTHER_CAPTIONS[name]}}}",
        rf"    \label{{fig:visual-{name.replace('_', '-')}-other}}",
        r"\end{figure*}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs("tables", exist_ok=True)
    for name, cfg in CATEGORIES.items():
        tex = build_figure(name, cfg)
        out_path = f"tables/visual_examples_{name}.tex"
        with open(out_path, "w") as f:
            f.write(tex + "\n")
        print(f"=== {out_path} ===")
        print(tex)
        print()

    # needs \usepackage{graphicx} and \usepackage{subcaption}; [p] lets all three
    # full-width figures share one float page
    tex = "\n\n".join(build_other_datasets_figure(name) for name in OTHER_SELECTION)
    out_path = "tables/visual_examples_other_datasets.tex"
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"=== {out_path} ===")
    print(tex)

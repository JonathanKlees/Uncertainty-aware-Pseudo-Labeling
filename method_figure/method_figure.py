#!/usr/bin/env python3
"""
Method overview figure for the LPOD pipeline — iterative query/label loop.

Stages (left to right, looping back at the end):
  A  Object Graph (unlabeled)     — similarity graph over OVOD region proposals
  Query           (arrow only)    — select one instance from the graph
  B  Queried Instance (unlabeled) — extracted region, class unknown
  Evaluation      (arrow only)    — assigns a class to the queried instance
  C  Evaluated Instance (labeled) — region with an assigned pseudo-label
  Label Propagation (arrow only)  — spreads the new label through the graph
  D  Object Graph (propagated)    — graph re-colored with soft/propagated labels
  Next Iteration  (dashed arrow, loops back under the row to A)
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── palette (matches utils/plot_layout.py) ─────────────────────────
# ACCENT_NAME picks the label color (the paper figure uses coral).
ACCENT_NAME   = "coral"
# per accent: (base color, white blend for ACCENT_MID, accent share in ACCENT_FAINT).
# Coral is much lighter than teal, so it needs less white and more accent
# in the grey blend to keep the propagated borders readable.
ACCENTS       = {"teal": ("#008080", 0.30, 0.35), "coral": ("#FF7F50", 0.12, 0.60)}
SLATEGRAY     = "#708090"


def blend_color(c1, c2, t):
    """Blend two '#rrggbb' colors; t=0 -> c1, t=1 -> c2."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


ACCENT, MID_WHITE, FAINT_SHARE = ACCENTS[ACCENT_NAME]
# lighter variant for propagated (soft) labels
ACCENT_MID    = blend_color(ACCENT, "#ffffff", MID_WHITE)


# giraffes only reachable through giraffe_prop_1's strong edges (i.e. two
# hops from the query node) get a fainter grey/accent blend than prop_1
# itself — a hint of propagation rather than a confident one.
ACCENT_FAINT = blend_color(SLATEGRAY, ACCENT_MID, FAINT_SHARE)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size"  : 9.0,
    "text.color" : "#222222",
})

# ── figure setup (wide, shallow — double-column paper width) ──────
fig, ax = plt.subplots(figsize=(15.4, 5.0))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 19.6)
ax.set_ylim(-1.5, 5.0)
ax.axis("off")
ax.set_aspect("equal")


def R(x, y, w, h, fc="none", ec="black", lw=1.0, ls="solid", alpha=1.0, z=3):
    ax.add_patch(Rectangle(
        (x, y), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        alpha=alpha, zorder=z,
    ))


def draw_panel(x0, y0, x1, y1, caption):
    cx = (x0 + x1) / 2
    ax.text(cx, y1 + 0.14, caption,
            ha="center", va="bottom", fontsize=10.5, color=SLATEGRAY, style="italic")


def arrow(xy_from, xy_to, ls="-", color=SLATEGRAY, lw=1.8, z=5):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, linestyle=ls),
        zorder=z,
    )


def add_crop(name, cx, cy, target_h, z=4):
    img = plt.imread(os.path.join(ASSETS_DIR, f"{name}.png"))
    h_px, w_px = img.shape[0], img.shape[1]
    aspect = w_px / h_px
    h = target_h
    w = h * aspect
    x0, y0 = cx - w / 2, cy - h / 2
    ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), zorder=z)
    return x0, y0, w, h


def add_node(name, cx, cy, target_h, state, label=None, lw_scale=1.0):
    """state: 'unlabeled' (dotted gray) | 'query' (solid accent, strong)
              | 'propagated' (solid accent, lighter)
              | 'propagated_weak' (solid accent, faint — soft/uncertain spread)
              | 'propagated_faint' (solid grey/accent blend — minimal spread)
    Returns the padded bounding box (x0, y0, x1, y1) of the drawn node,
    i.e. including its border, for content-aware layout of arrows/gaps.
    """
    x0, y0, w, h = add_crop(name, cx, cy, target_h, z=4)
    pad = 0.04 * target_h
    if state == "unlabeled":
        ec, lw, ls, a = SLATEGRAY, 1.1, (0, (2, 1.5)), 0.75
    elif state == "query":
        ec, lw, ls, a = ACCENT, 2.2, "solid", 0.95
    elif state == "propagated_weak":
        ec, lw, ls, a = ACCENT_MID, 1.1, "solid", 0.42
    elif state == "propagated_faint":
        ec, lw, ls, a = ACCENT_FAINT, 1.1, "solid", 0.75
    else:  # propagated
        ec, lw, ls, a = ACCENT_MID, 1.4, "solid", 0.65
    lw *= lw_scale
    R(x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad,
      ec=ec, lw=lw, ls=ls, alpha=a, z=5)
    if label:
        ax.text(cx, y0 + h + pad + 0.05, label,
                ha="center", va="bottom", fontsize=10.0, fontweight="bold",
                color=ec, alpha=min(a + 0.15, 1.0), zorder=6)
    return x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad


# ── panel bounds  [x0, y0, x1, y1] — single row, left to right ────
# Gaps are tuned per transition: the graph panels (A, D) fill almost their
# whole box with content, while the single-node panels (B, C) leave a lot
# of empty margin around their (fixed-height) crop — so equal box-to-box
# gaps read as very uneven whitespace. gap_bc is narrower (both sides
# already have margin) and gap_cd is wider (both sides are tight against
# content, and it has to fit the longest arrow label, "Label Propagation").
panel_w, panel_h = 3.3, 3.0
# The graph panels (A, D) are scaled up uniformly (box + nodes together,
# see NODE_TARGET_H_GRAPH below) so the proposal crops read better —
# without changing the graph's internal layout, just its size. They're
# bottom-extended rather than centered so all four captions stay
# top-aligned in one row.
GRAPH_SCALE = 1.35
graph_w, graph_h = panel_w * GRAPH_SCALE, panel_h * GRAPH_SCALE
gap_ab, gap_bc, gap_cd = 1.15, 0.85, 1.45
y1 = 1.05 + panel_h
y0 = y1 - panel_h
y0_graph = y1 - graph_h

A = [0.30, y0_graph, 0.30 + graph_w, y1]
B = [A[2] + gap_ab, y0, A[2] + gap_ab + panel_w, y1]
C = [B[2] + gap_bc, y0, B[2] + gap_bc + panel_w, y1]
D = [C[2] + gap_cd, y0_graph, C[2] + gap_cd + graph_w, y1]

draw_panel(*A, "Object Graph: Unlabeled Region Proposals")
draw_panel(*B, "Queried Instance")
draw_panel(*C, "Manual Evaluation: Class Assignment")
draw_panel(*D, "Label Information Propagates")

# ════════════════════════════════════════════════════════════════════
# GRAPH DATA (shared node layout for panels A and D)
# ════════════════════════════════════════════════════════════════════
QUERY_NODE = "zebra_prop_4"

# node: (rx, ry) relative position within panel [0,1] x [0,1]
NODE_POS = {
    "zebra_gt_1":     (0.18, 0.22),
    "zebra_prop_1":   (0.40, 0.15),
    "zebra_prop_2":   (0.10, 0.52),
    "zebra_prop_4":   (0.34, 0.46),
    "giraffe_gt_1":   (0.82, 0.80),
    "giraffe_prop_1": (0.62, 0.60),
    "giraffe_prop_4": (0.90, 0.48),
    "giraffe_prop_5": (0.68, 0.86),
    # false-positive proposals (background clutter) — grouped in their own
    # top-left cluster, connected to each other but isolated from the real
    # object clusters, never labeled
    "fp_tree_1":      (0.08, 0.90),
    "fp_tree_2":      (0.28, 0.78),
}
FP_NODES = {"fp_tree_1", "fp_tree_2"}
FP_EDGES = [("fp_tree_1", "fp_tree_2")]
STRONG_EDGES = [
    ("zebra_gt_1", "zebra_prop_1"), ("zebra_gt_1", "zebra_prop_4"),
    ("zebra_prop_1", "zebra_prop_4"), ("zebra_prop_2", "zebra_prop_4"),
    ("zebra_prop_2", "zebra_gt_1"),
    ("giraffe_gt_1", "giraffe_prop_1"), ("giraffe_gt_1", "giraffe_prop_5"),
    ("giraffe_prop_1", "giraffe_prop_4"), ("giraffe_prop_4", "giraffe_prop_5"),
]
WEAK_EDGES = [("zebra_prop_4", "giraffe_prop_1")]

NODE_TARGET_H = 0.62 * GRAPH_SCALE


def panel_xy(panel, rx, ry):
    x0, y0, x1, y1 = panel
    return x0 + rx * (x1 - x0), y0 + ry * (y1 - y0)


def draw_graph_edges(panel, edge_color=SLATEGRAY):
    for n1, n2 in STRONG_EDGES + FP_EDGES:
        x1, y1 = panel_xy(panel, *NODE_POS[n1])
        x2, y2 = panel_xy(panel, *NODE_POS[n2])
        ax.plot([x1, x2], [y1, y2], color=edge_color, lw=4.0, alpha=0.6, zorder=2)
    for n1, n2 in WEAK_EDGES:
        x1, y1 = panel_xy(panel, *NODE_POS[n1])
        x2, y2 = panel_xy(panel, *NODE_POS[n2])
        ax.plot([x1, x2], [y1, y2], color=SLATEGRAY, lw=2.2, ls="solid",
                alpha=0.48, zorder=2)


# ════════════════════════════════════════════════════════════════════
# PANEL A — Object Graph (unlabeled)
# ════════════════════════════════════════════════════════════════════
draw_graph_edges(A)
content_right_A = A[0]
for name, (rx, ry) in NODE_POS.items():
    cx, cy = panel_xy(A, rx, ry)
    _, _, nx1, _ = add_node(name, cx, cy, NODE_TARGET_H, state="unlabeled")
    content_right_A = max(content_right_A, nx1)


# ════════════════════════════════════════════════════════════════════
# PANEL B — Queried Instance (unlabeled, extracted)
# ════════════════════════════════════════════════════════════════════
bx, by = (B[0] + B[2]) / 2, (B[1] + B[3]) / 2 - 0.10
content_left_B, _, content_right_B, _ = add_node(
    QUERY_NODE, bx, by, 1.75, state="unlabeled", label="class: ?")


# ════════════════════════════════════════════════════════════════════
# PANEL C — Evaluated Instance (labeled)
# ════════════════════════════════════════════════════════════════════
cx_, cy_ = (C[0] + C[2]) / 2, (C[1] + C[3]) / 2 - 0.10
content_left_C, _, content_right_C, _ = add_node(
    QUERY_NODE, cx_, cy_, 1.75, state="query", label="zebra")


# ════════════════════════════════════════════════════════════════════
# PANEL D — Object Graph (propagated)
# ════════════════════════════════════════════════════════════════════
draw_graph_edges(D)
content_left_D = D[2]
for name, (rx, ry) in NODE_POS.items():
    cx, cy = panel_xy(D, rx, ry)
    if name == QUERY_NODE:
        state = "query"
    elif name in FP_NODES:
        state = "unlabeled"  # false positives never receive a label
    elif name.startswith("zebra"):
        state = "propagated"
    elif name == "giraffe_prop_1":
        # weakly connected to the query node -> soft, uncertain propagation
        state = "propagated_weak"
    elif name.startswith("giraffe"):
        # two hops from the query (via giraffe_prop_1) -> minimal feedback
        state = "propagated_faint"
    else:
        state = "unlabeled"
    nx0, _, _, _ = add_node(name, cx, cy, NODE_TARGET_H, state=state, lw_scale=1.5)
    content_left_D = min(content_left_D, nx0)


# ════════════════════════════════════════════════════════════════════
# INTER-PANEL ARROWS  (plain arrows + text labels, no module boxes)
# ════════════════════════════════════════════════════════════════════
# Endpoints are anchored to each panel's actual drawn content (not its
# nominal box) so the arrow — and its label — sit centered in the real
# whitespace, regardless of how much padding a panel's content leaves.
mid_y = (y0 + y1) / 2
margin = 0.12

for (x_from, x_to, txt) in [
    (content_right_A, content_left_B, "Query"),
    (content_right_B, content_left_C, "Evaluation"),
    (content_right_C, content_left_D, "Label\nPropagation"),
]:
    arrow((x_from + margin, mid_y), (x_to - margin, mid_y), color=SLATEGRAY, lw=1.8)
    ax.text((x_from + x_to) / 2, mid_y + 0.13, txt,
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1A1A1A")

# loop-back: D -> A, dashed, arcing under the bottom of the row.
# Extra vertical drop gives the short up-arrow into A room to render a
# proper-looking head instead of a stubby tip crowded by the dash pattern.
bottom_y = y0_graph - 0.62
xA_c, xD_c = (A[0] + A[2]) / 2, (D[0] + D[2]) / 2
ax.plot([xD_c, xD_c], [y0_graph - 0.05, bottom_y], color=SLATEGRAY, lw=1.6, ls=(0, (4, 2)), zorder=5)
ax.plot([xD_c, xA_c], [bottom_y, bottom_y], color=SLATEGRAY, lw=1.6, ls=(0, (4, 2)), zorder=5)
arrow((xA_c, bottom_y), (xA_c, y0_graph - 0.05), ls=(0, (4, 2)), color=SLATEGRAY, lw=1.6)
ax.text((xA_c + xD_c) / 2, bottom_y + 0.10, "Next Iteration",
        ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=SLATEGRAY)


plt.savefig(os.path.join(OUT_DIR, "method_figure.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT_DIR, "method_figure.pdf"), dpi=300,
            bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("Saved method_figure.png and method_figure.pdf")

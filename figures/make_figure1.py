#!/usr/bin/env python3
"""
make_figure1.py — Figure 1 (methodology: architecture and evaluation protocol).

The video and spectrogram thumbnails are cropped from the dataset sample figure in the
results package (Figure_40A_Refined_Actual_FakeAVCeleb_AV_Samples.png).

Usage:  python make_figure1.py --results <SMDNet_V2_All_Publication_Results> --out figures_out
Works in a script or pasted into a notebook cell (set RESULTS_DIR / OUTPUT_DIR below).
"""
import argparse, os
from pathlib import Path
RESULTS_DIR = "/content/SMDNet_V2_All_Publication_Results"
OUTPUT_DIR = "/content/figures_out"
_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=os.environ.get("SMDNET_RESULTS", RESULTS_DIR))
_ap.add_argument("--out", default=os.environ.get("SMDNET_FIG_OUT", OUTPUT_DIR))
_args, _ = _ap.parse_known_args()
OUT = Path(_args.out); OUT.mkdir(parents=True, exist_ok=True)
_src = None
for cand in [Path(_args.results)] + ([p for p in Path(_args.results).iterdir() if p.is_dir()] if Path(_args.results).is_dir() else []):
    f = cand / "01_Figures/Final_Publication_Figures/Dataset_Visualization/Figure_40A_Refined_Actual_FakeAVCeleb_AV_Samples.png"
    if f.is_file(): _src = f; break
if _src is None:
    raise SystemExit("Sample figure not found: set --results to the unzipped SMDNet_V2_All_Publication_Results folder.")
CROPS_DIR = OUT / "_crops"; CROPS_DIR.mkdir(exist_ok=True)
from PIL import Image as _Image
_im = _Image.open(_src).convert("RGB")
# pixel boxes of the frames (3 per category) and spectrograms in the 4729 x 3189 sample figure
for _code, _y in {"RVRA": 126, "RVFA": 908, "FVRA": 1690, "FVFA": 2472}.items():
    for _j, _x in enumerate([146, 1034, 1921]):
        _im.crop((_x + 4, _y + 4, _x + 575, _y + 575)).save(CROPS_DIR / f"{_code}_f{_j + 1}.png")
    _im.crop((2745, _y + 95, 4282, _y + 566)).save(CROPS_DIR / f"{_code}_spec.png")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon,
                                Circle, Arc, Wedge)
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
CROPS = str(CROPS_DIR) + "/"
W, H = 180, 137
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

INK = "#2B2B2B"; MUTED = "#6B6B6B"
BLUE, BLUE_L = "#2C6BAC", "#DCE8F5"
ORNG, ORNG_L = "#C4621D", "#FBE7D6"
GRN, GRN_L = "#2E7D4F", "#E1F0E5"
PURP, PURP_L = "#6A4BB5", "#ECE6F7"
RED, RED_L = "#B23A3A", "#F8E3E3"
REAL, FAKE = "#2E7D4F", "#B23A3A"

def T(x, y, s, fs=6.2, c=INK, w="normal", ha="center", va="center", **k):
    k.setdefault("zorder", 10)
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va, **k)

def panel_bg(x, y, w, h, fc="#F7F8FA"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                                fc=fc, ec="none", zorder=0))

def arrow(p, q, c=INK, lw=0.75, dashed=False, ms=6, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                 ls=(0, (3, 2)) if dashed else "-", connectionstyle=conn,
                                 shrinkA=0, shrinkB=0, zorder=5))

def img(path, x, y, w, h, border="white", lw=0.6, z=3):
    ax.imshow(np.asarray(Image.open(path)), extent=(x, x + w, y, y + h), aspect="auto",
              zorder=z, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=border, lw=lw, zorder=z + 0.1))

def shadow_rect(x, y, w, h, z=2):
    ax.add_patch(Rectangle((x + 0.5, y - 0.5), w, h, fc="#00000022", ec="none", zorder=z))

# ---------------------------------------------------------------- icons
def cube(x, y, w, h, d, fc, ec, z=4):
    """3-D block: front face at (x, y), depth d toward upper-right."""
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True,
                         fc=fc, ec=ec, lw=0.55, zorder=z))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
                         closed=True, fc=lighten(fc, 0.35), ec=ec, lw=0.55, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
                         closed=True, fc=darken(fc, 0.12), ec=ec, lw=0.55, zorder=z))

def lighten(c, f):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(c + (1 - c) * f)

def darken(c, f):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(c * (1 - f))

def conv_stack(x, y, specs, fc, ec, gap=1.2):
    """specs: list of (w, h, d); blocks bottom-aligned on a centre line."""
    cx = x
    for w, h, d in specs:
        cube(cx, y - h / 2, w, h, d, fc, ec)
        cx += w + gap
    return cx

def transformer_icon(x, y, w, h, c):
    n = 5; s = 2.2; xs = np.linspace(x + 1.5, x + w - 1.5 - s, n)
    yb, yt = y + 1.2, y + h - 1.2 - s
    for xi in xs:
        for xj in xs:
            ax.plot([xi + s / 2, xj + s / 2], [yb + s, yt], color=c, lw=0.3, alpha=0.55, zorder=4)
    for xi in xs:
        ax.add_patch(Rectangle((xi, yb), s, s, fc=lighten(c, 0.55), ec=c, lw=0.5, zorder=5))
        ax.add_patch(Rectangle((xi, yt), s, s, fc=c, ec=c, lw=0.5, zorder=5))

def funnel(x, y, w, h, c):
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h * 0.66), (x + w, y + h * 0.34), (x, y)],
                         closed=True, fc=lighten(c, 0.6), ec=c, lw=0.55, zorder=4))

def vector(x, y, n, cw, ch, fc, ec, vertical=True):
    for i in range(n):
        if vertical:
            ax.add_patch(Rectangle((x, y + i * ch), cw, ch, fc=fc, ec=ec, lw=0.45, zorder=5))
        else:
            ax.add_patch(Rectangle((x + i * cw, y), cw, ch, fc=fc, ec=ec, lw=0.45, zorder=5))

def lstm_icon(x, y, c):
    n, r, sp = 4, 1.35, 4.2
    for row, dy in [(0, 2.4), (1, -2.4)]:
        cxs = [x + r + i * sp for i in range(n)]
        for i, cx in enumerate(cxs):
            ax.add_patch(Circle((cx, y + dy), r, fc=lighten(c, 0.55), ec=c, lw=0.55, zorder=5))
            if i < n - 1:
                a, b = (cx + r, y + dy), (cxs[i + 1] - r, y + dy)
                if row == 1: a, b = b, a
                arrow(a, b, c=c, lw=0.5, ms=3.5)
    return x + 2 * r + (n - 1) * sp

def mlp_icon(x, y, c, layers=(4, 3, 1), sp=4.6, vs=2.6):
    pos = []
    for li, n in enumerate(layers):
        ys = [y + (i - (n - 1) / 2) * vs for i in range(n)]
        pos.append([(x + li * sp, yy) for yy in ys])
    for a, b in zip(pos, pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=0.3, alpha=0.6, zorder=4)
    for layer in pos:
        for p in layer:
            ax.add_patch(Circle(p, 0.85, fc="white", ec=c, lw=0.55, zorder=5))
    return pos[-1][0][0]

def sigmoid_icon(x, y, w, h, c):
    t = np.linspace(-5, 5, 60)
    ax.plot(x + (t + 5) / 10 * w, y + h / (1 + np.exp(-t * 1.2)), color=c, lw=0.9, zorder=5)
    ax.plot([x, x + w], [y, y], color=MUTED, lw=0.35, zorder=4)
    ax.plot([x, x], [y, y + h], color=MUTED, lw=0.35, zorder=4)

def sim_matrix(x, y, n, s, c):
    for i in range(n):
        for j in range(n):
            fc = c if i == j else lighten(c, 0.82)
            ax.add_patch(Rectangle((x + j * s, y + (n - 1 - i) * s), s, s, fc=fc, ec="white", lw=0.4, zorder=5))

def waveform(x, y, w, h, c, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 400)
    env = 0.25 + 0.75 * np.abs(np.sin(t * np.pi * 3.2)) ** 1.5
    sig = env * rng.normal(0, 0.45, t.size)
    sig = np.clip(sig, -1, 1)
    ax.fill_between(x + t * w, y + h / 2 - sig * h / 2, y + h / 2 + sig * h / 2, color=c, lw=0, zorder=4)

def person(x, y, s, c, z=5):
    ax.add_patch(Circle((x, y + s * 0.62), s * 0.22, fc=c, ec="none", zorder=z))
    ax.add_patch(Wedge((x, y), s * 0.42, 0, 180, fc=c, ec="none", zorder=z))

def file_icon(x, y, w, h, c, label):
    f = 1.6
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)],
                         closed=True, fc="white", ec=c, lw=0.6, zorder=5))
    ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)],
                         closed=True, fc=lighten(c, 0.6), ec=c, lw=0.5, zorder=6))
    T(x + w / 2, y + h * 0.42, label, fs=4.6, c=c, w="bold", zorder=7)

def lock_icon(x, y, s, c):
    ax.add_patch(Arc((x + s / 2, y + s * 0.62), s * 0.6, s * 0.7, theta1=0, theta2=180, color=c, lw=0.9, zorder=5))
    ax.add_patch(FancyBboxPatch((x, y), s, s * 0.62, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=c, ec="none", zorder=6))
    ax.add_patch(Circle((x + s / 2, y + s * 0.32), s * 0.09, fc="white", ec="none", zorder=7))

def scale_icon(x, y, s, c):
    ax.plot([x + s / 2, x + s / 2], [y, y + s * 0.8], color=c, lw=0.9, zorder=5)
    ax.plot([x, x + s], [y + s * 0.8, y + s * 0.8], color=c, lw=0.9, zorder=5)
    for px in (x, x + s):
        ax.add_patch(Wedge((px, y + s * 0.42), s * 0.22, 180, 360, fc=lighten(c, 0.45), ec=c, lw=0.5, zorder=5))
        ax.plot([px - s * 0.2, px, px + s * 0.2], [y + s * 0.42, y + s * 0.8, y + s * 0.42], color=c, lw=0.4, zorder=4)
    ax.add_patch(Polygon([(x + s * 0.32, y), (x + s * 0.68, y), (x + s / 2, y + s * 0.16)],
                         closed=True, fc=c, ec="none", zorder=5))

def slider_icon(x, y, w, c, pos=0.62):
    ax.plot([x, x + w], [y, y], color=lighten(c, 0.4), lw=1.6, solid_capstyle="round", zorder=5)
    ax.plot([x, x + w * pos], [y, y], color=c, lw=1.6, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((x + w * pos, y), 1.1, fc="white", ec=c, lw=0.8, zorder=6))

def film_icon(x, y, w, h, c):
    ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="none", zorder=5))
    for i in range(5):
        for yy in (y + 0.45, y + h - 1.15):
            ax.add_patch(Rectangle((x + 0.6 + i * (w - 1.2) / 5, yy), 0.75, 0.7, fc="white", ec="none", zorder=6))
    ax.add_patch(Rectangle((x + 0.6, y + 1.6), w - 1.2, h - 3.2, fc=lighten(c, 0.75), ec="none", zorder=6))

def check(x, y, ok, s=1.25):
    c = REAL if ok else FAKE
    ax.add_patch(Circle((x, y), s, fc=c, ec="none", zorder=7))
    T(x, y - 0.05, "✓" if ok else "✗", fs=4.8, c="white", w="bold", zorder=8)

def tag(x, y, kind, ok, w=9.2, h=3.9):
    c = REAL if ok else FAKE
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc=lighten(c, 0.86), ec=lighten(c, 0.25), lw=0.5, zorder=5))
    T(x + w / 2, y + h / 2 - 0.05, f"{kind} {'real' if ok else 'fake'}", fs=4.6, c=c, zorder=6)

def label(x, y, letter, title):
    T(x, y, letter, fs=10, w="bold", ha="left")
    T(x + 5, y, title, fs=7.4, w="bold", ha="left")

# =============================================================================
# b  ARCHITECTURE
# =============================================================================
label(2, 133, "a", "SMD-Net V2 architecture (28.4 M parameters)")
panel_bg(2, 62, 176, 67)
yv, ya = 112, 80            # branch centre lines
T(4, yv + 12.5, "Visual branch", fs=5.8, c=BLUE, w="bold", ha="left")
T(4, ya - 13.0, "Audio branch", fs=5.8, c=ORNG, w="bold", ha="left")

# --- visual input: stacked frames
for k, dx in enumerate([3.0, 1.5, 0.0]):
    shadow_rect(4 + dx, yv - 6 + dx * 0.6, 12, 12)
    img(CROPS + f"FVRA_f{3 - k}.png", 4 + dx, yv - 6 + dx * 0.6, 12, 12, border="#D9D9D9", lw=0.35)
T(11, yv - 9.3, "16 frames, 0.8 s\n224 × 224", fs=5.0, c=MUTED, linespacing=1.25)
arrow((20, yv), (24, yv))
# --- ResNet-18
end = conv_stack(25, yv, [(2.2, 13, 2.2), (2.2, 10.5, 2.0), (2.2, 8, 1.8), (2.2, 6, 1.5)], BLUE_L, BLUE, gap=1.0)
T(31.5, yv - 10.3, "ResNet-18\nper frame", fs=5.4, linespacing=1.25)
arrow((end + 2.2, yv), (end + 5.6, yv))
# --- Transformer
tx = end + 6.4
ax.add_patch(FancyBboxPatch((tx, yv - 6.5), 17, 13, boxstyle="round,pad=0,rounding_size=1",
                            fc="white", ec=BLUE, lw=0.6, zorder=3))
transformer_icon(tx, yv - 6.5, 17, 13, BLUE)
T(tx + 8.5, yv - 10.3, "Transformer\nencoder × 2", fs=5.4, linespacing=1.25)
arrow((tx + 17.5, yv), (tx + 20.5, yv))
# --- pool + vector
px = tx + 21.2
funnel(px, yv - 5, 5, 10, BLUE)
vector(px + 7, yv - 6, 6, 1.8, 2.0, BLUE, "white")
T(px + 4.5, yv - 10.3, "mean pool\nv ∈ ℝ⁵¹²", fs=5.4, linespacing=1.25)

# --- audio input: waveform -> mel
ax.add_patch(FancyBboxPatch((4, ya - 5), 13, 10, boxstyle="round,pad=0,rounding_size=1",
                            fc="white", ec=ORNG, lw=0.6, zorder=3))
waveform(5, ya - 4, 11, 8, ORNG, seed=7)
T(10.5, ya - 8.6, "16 kHz mono\n12,800 samples", fs=5.0, c=MUTED, linespacing=1.25)
arrow((18, ya), (21, ya))
shadow_rect(21.6, ya - 5.5, 15, 11)
img(CROPS + "FVRA_spec.png", 21.6, ya - 5.5, 15, 11)
T(29.1, ya - 9.1, "log-mel\n128 × 81", fs=5.4, linespacing=1.25)
arrow((37.4, ya), (40.4, ya))
# --- CNN
end_a = conv_stack(41, ya, [(2.2, 11, 2.0), (2.2, 8.5, 1.8), (2.2, 6.5, 1.5)], ORNG_L, ORNG, gap=1.0)
T(45.5, ya - 9.1, "CNN", fs=5.4)
arrow((end_a + 1.8, ya), (end_a + 4.6, ya))
# --- BiLSTM
lx = end_a + 5.4
lend = lstm_icon(lx, ya, ORNG)
T((lx + lend) / 2, ya - 9.1, "BiLSTM", fs=5.4)
arrow((lend + 0.8, ya), (lend + 3.6, ya))
pxa = lend + 4.3
funnel(pxa, ya - 5, 5, 10, ORNG)
vector(pxa + 7, ya - 6, 6, 1.8, 2.0, ORNG, "white")
T(pxa + 4.5, ya - 9.1, "mean pool\na ∈ ℝ⁵¹²", fs=5.4, linespacing=1.25)

# --- fusion
vx_end = px + 8.8; ax_end = pxa + 8.8
fx = 112
arrow((vx_end + 0.6, yv), (fx, 100.2), conn="arc3,rad=-0.15")
arrow((ax_end + 0.6, ya), (fx, 91.8), conn="arc3,rad=0.15")
vector(fx + 0.6, 96, 6, 1.6, 1.25, BLUE, "white")
vector(fx + 0.6, 88.4, 6, 1.6, 1.25, ORNG, "white")
T(fx + 1.4, 84.6, "concat\n1,024", fs=5.0, c=MUTED, linespacing=1.2)
arrow((fx + 3.2, 96), (fx + 6.4, 96))
mend = mlp_icon(fx + 7.6, 96, GRN)
T(fx + 14.2, 104.6, "MLP\n1,024 → 512 → 1", fs=5.4, linespacing=1.25)
arrow((mend + 1.4, 96), (mend + 4.4, 96))
sigmoid_icon(mend + 5.0, 92.2, 7.5, 7.5, GRN)
T(mend + 8.75, 88.0, "sigmoid", fs=5.4)
# --- video score with 5 windows
sx = mend + 15.5
arrow((sx - 2.2, 96), (sx + 0.6, 96))
ax.add_patch(FancyBboxPatch((sx + 1, 91), 21, 10, boxstyle="round,pad=0,rounding_size=1.2",
                            fc=GRN_L, ec=GRN, lw=0.6, zorder=3))
ax.plot([sx + 3, sx + 20], [96.2, 96.2], color=GRN, lw=0.6, zorder=4)
for f in [0, .25, .5, .75, 1]:
    xx = sx + 3 + 17 * f
    ax.add_patch(Rectangle((xx - 0.9, 95.1), 1.8, 2.2, fc=GRN, ec="white", lw=0.3, zorder=5))
T(sx + 11.5, 99.3, "5 windows → mean", fs=5.0, c=GRN, w="bold")
T(sx + 11.5, 93.0, "p(manipulated)", fs=5.4)

# --- Stage-A heads (dashed)
hx = 125
ax.add_patch(Polygon([(hx, yv + 3.5), (hx + 7, yv + 1.8), (hx + 7, yv - 1.8), (hx, yv - 3.5)],
                     closed=True, fc=PURP_L, ec=PURP, lw=0.55, ls=(0, (2.5, 1.5)), zorder=4))
ax.add_patch(Polygon([(hx, ya + 3.5), (hx + 7, ya + 1.8), (hx + 7, ya - 1.8), (hx, ya - 3.5)],
                     closed=True, fc=PURP_L, ec=PURP, lw=0.55, ls=(0, (2.5, 1.5)), zorder=4))
arrow((vx_end + 0.6, yv + 2.5), (hx, yv + 1.5), c=PURP, dashed=True)
arrow((ax_end + 0.6, ya - 2.5), (hx, ya - 1.5), c=PURP, dashed=True)
T(hx + 3.5, yv + 6.2, "proj. 128-d", fs=5.0, c=PURP)
T(hx + 3.5, ya - 6.0, "proj. 128-d", fs=5.0, c=PURP)
smx = 150
sim_matrix(smx, 105.5, 5, 2.4, PURP)
arrow((hx + 7.5, 112.7), (smx - 0.8, 112.7), c=PURP, dashed=True)
ax.plot([hx + 7.5, 172.5, 172.5], [ya, ya, 110.3], color=PURP, lw=0.75, ls=(0, (3, 2)), zorder=4)
arrow((172.5, 110.3), (162.8, 110.3), c=PURP, dashed=True)
T(smx + 6, 124.6, "InfoNCE, τ = 0.07", fs=5.4, c=PURP, w="bold")
T(smx + 6, 121.2, "Stage A only", fs=5.0, c=PURP, style="italic")

# =============================================================================
# c  PROTOCOL + AUDITS
# =============================================================================
label(2, 56.5, "b", "Frozen training and evaluation protocol, and post-hoc audits")
panel_bg(2, 3, 176, 49)

# split bar
bx, by, bw = 6, 40.5, 34
tot = 21476
segs = [("train", 15065, "#9DB4CC"), ("val", 3237, "#C9B37E"), ("test", 3174, "#C98F8F")]
cx = bx
for name, n, c in segs:
    w = bw * n / tot
    ax.add_patch(Rectangle((cx, by), w, 4.2, fc=c, ec="white", lw=0.6, zorder=4))
    cx += w
T(bx, by + 7.2, "Source-identity-disjoint split", fs=5.6, w="bold", ha="left")
T(bx, by - 2.6, "train 15,065 · val 3,237", fs=5.0, c=MUTED, ha="left")
T(bx, by - 5.6, "test 3,174 · seed 42", fs=5.0, c=MUTED, ha="left")

steps = [(56, "Stage A", "contrastive SSL\n331 authentic videos", PURP, "sim"),
         (85, "Stage B", "BCE, class-balanced\nsampler", GRN, "scale"),
         (114, "Threshold", "fixed on validation\nonly", "#7A6A2E", "slider"),
         (143, "Test once", "untouched test set\n5-window mean", INK, "lock")]
for i, (x, t1, t2, c, kind) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x - 11, 33.2), 22, 15.5, boxstyle="round,pad=0,rounding_size=1.4",
                                fc="white", ec=lighten(c, 0.3), lw=0.6, zorder=3))
    if kind == "sim": sim_matrix(x - 3.2, 41.4, 4, 1.6, PURP)
    if kind == "scale": scale_icon(x - 3.2, 41.6, 6.4, GRN)
    if kind == "slider": slider_icon(x - 4.5, 44.2, 9, "#7A6A2E")
    if kind == "lock": lock_icon(x - 2.4, 41.4, 4.8, INK)
    T(x, 39.0, t1, fs=5.8, w="bold", c=c)
    T(x, 35.6, t2, fs=4.9, c=MUTED, linespacing=1.2)
    if i < 3:
        arrow((x + 11.4, 41), (x + 17.6, 41))
arrow((bx + bw + 0.8, 42.6), (44.6, 42.6))

# audits
T(6, 29.9, "Post-hoc audits of the frozen predictions", fs=5.8, w="bold", c=RED, ha="left")
ax.plot([143, 143], [33.0, 27.4], color=RED, lw=0.6, zorder=2)
ax.plot([5 + 41 / 2, 5 + 3 * 43.6 + 41 / 2], [27.4, 27.4], color=RED, lw=0.6, zorder=2)
audits = [
    ("Encoding metadata", "codec, sample rate\nand fps vs. label", "meta"),
    ("Identity leakage", "source and target\nidentities shared\nacross splits", "id"),
    ("Codec control", "remux; re-encode\nvideo, audio or both;\nCRF sweep", "codec"),
    ("Reproducibility", "hashes, provenance,\nAMP re-inference", "hash"),
]
aw = 41
for i, (t1, t2, kind) in enumerate(audits):
    x = 5 + i * (aw + 2.6)
    ax.add_patch(FancyBboxPatch((x, 6), aw, 17.5, boxstyle="round,pad=0,rounding_size=1.4",
                                fc="white", ec=lighten(RED, 0.45), lw=0.6, zorder=3))
    ix, iy = x + 4, 13
    if kind == "meta":
        file_icon(ix - 1.5, iy - 1, 6.5, 8.2, REAL, "H.264")
        file_icon(ix + 6.0, iy - 1, 6.5, 8.2, FAKE, "MP4V")
    if kind == "id":
        person(ix + 1.5, iy, 5.2, "#7E95AE"); person(ix + 5.6, iy, 5.2, "#B98C8C")
        T(ix + 3.6, iy - 2.6, "train ∩ test", fs=4.4, c=MUTED)
    if kind == "codec":
        film_icon(ix - 1.5, iy + 1, 6, 4.6, FAKE)
        arrow((ix + 4.9, iy + 3.3), (ix + 7.6, iy + 3.3), ms=4)
        film_icon(ix + 8.0, iy + 1, 6, 4.6, REAL)
        T(ix + 6.2, iy - 1.6, "H.264 CRF 18", fs=4.4, c=MUTED)
    if kind == "hash":
        ax.add_patch(Circle((ix + 3.5, iy + 3.2), 3.6, fc=lighten(INK, 0.85), ec=INK, lw=0.5, zorder=5))
        T(ix + 3.5, iy + 3.0, "#", fs=9, w="bold", zorder=6)
        T(ix + 3.5, iy - 1.8, "SHA-256", fs=4.4, c=MUTED)
    tx_ = x + 18.5
    T(tx_, 19.2, t1, fs=5.6, w="bold", ha="left")
    T(tx_, 15.6, t2, fs=4.9, c=MUTED, ha="left", va="top", linespacing=1.3)
    arrow((x + aw / 2, 27.4), (x + aw / 2, 23.6), c=RED, ms=4)

for ext, kw in [("pdf", {}), ("png", {"dpi": 600}),
                ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
    fig.savefig(str(OUT / f"Figure1_Methodology.{ext}"), **kw)
print("saved")

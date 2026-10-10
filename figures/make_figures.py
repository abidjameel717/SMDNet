#!/usr/bin/env python3
"""
make_figures.py — regenerate Figures 2–10 of the SMD-Net V2 manuscript.

    Fig. 2  Sample audiovisual inputs (frames + log-mel spectrograms)
    Fig. 3  Cohort curation flow chart and dataset distribution
    Fig. 4  Training progress (Stage B ROC-AUC, Stage A InfoNCE)
    Fig. 5  ROC curves on the frozen test set
    Fig. 6  Confusion matrices: original test set and H.264 re-encoded test set
    Fig. 7  Ablation and error analysis
    Fig. 8  Encoding shortcut and codec-control experiments
    Fig. 9  Full-test-set codec normalisation
    Fig. 10 Protocol-matched comparison

Inputs (set with command-line options or environment variables):
    --results   unzipped SMDNet_V2_All_Publication_Results folder       (SMDNET_RESULTS)
    --project   Drive project folder SMDNet_V2_ScientificReports         (SMDNET_PROJECT)
                  05_Predictions/                      -> Fig. 5
                  08_Representative_Dataset_Samples/   -> Fig. 2
                  12_Full_Test_Codec_Normalization/    -> Figs. 6, 9, 10
    --notebook  development notebook (Cell 53 printed output)          (SMDNET_NOTEBOOK)
                                                         -> Fig. 8c,d
    --out       output folder (default: figures_out)
Usage:
    python make_figures.py                     # all figures
    python make_figures.py --figures 6 9       # selected figures

Each figure is written as PDF (vector), PNG and TIFF (600 dpi) at 180 mm width.
If an optional input is missing, the script says so and uses the values verified
for the manuscript instead, so every figure can still be produced.

Requirements: numpy, pandas, matplotlib, pillow; opencv-python and ffmpeg only for Fig. 2.
"""
import argparse, json, os, re, shutil, subprocess, tempfile, warnings

# =============================================================================
# SETTINGS FOR NOTEBOOK USE — edit these if you paste this script into a Colab cell.
# (Command-line options or environment variables override them.)
# =============================================================================
RESULTS_DIR  = "/content/SMDNet_V2_All_Publication_Results"          # unzipped results package
PROJECT_DIR  = "/content/drive/MyDrive/SMDNet_V2_ScientificReports"  # Drive project folder
NOTEBOOK     = "/content/drive/MyDrive/SMDNet_V2_ScientificReports/Video_forgery (1).ipynb"
OUTPUT_DIR   = "/content/figures_out"
FIGURES      = [2, 3, 4, 5, 6, 7, 8, 9, 10]
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED, GRID = "#2B2B2B", "#6B6B6B", "#E6E6E6"
REAL, FAKE = "#2E7D4F", "#B23A3A"
COL = {"Full_SSL": "#6A4BB5", "NoSSL": "#2E7D4F", "Visual_Only": "#2C6BAC"}
NAME = {"Full_SSL": "Full SSL", "NoSSL": "No SSL", "Visual_Only": "Visual only"}
CATS = [("RealVideo-RealAudio", "RVRA"), ("RealVideo-FakeAudio", "RVFA"),
        ("FakeVideo-RealAudio", "FVRA"), ("FakeVideo-FakeAudio", "FVFA")]
THR = {"Full_SSL": 0.75732421875, "NoSSL": 0.556396484375, "Visual_Only": 0.9638671875}
FW = 180  # figure width in mm

# =============================================================================
# Values verified for the manuscript (used only when an input file is missing)
# =============================================================================
VERIFIED_CONFUSION_ORIGINAL = {"Full_SSL": (72, 0, 7, 3095), "NoSSL": (72, 0, 1, 3101),
                               "Visual_Only": (72, 0, 63, 3039)}          # TN, FP, FN, TP
VERIFIED_CELL56 = {  # frozen thresholds; Cell 56 output
    "Full_SSL": {"orig": dict(balanced_accuracy=0.9989, macro_f1=0.9763, mcc=0.9536, roc_auc=1.0, tn=72, fp=0, fn=7, tp=3095),
                 "norm": dict(balanced_accuracy=0.8864, macro_f1=0.5207, mcc=0.2676, roc_auc=0.9747, tn=72, fp=0, fn=705, tp=2397),
                 "cat_correct": {"RVRA": (72, 72), "RVFA": (70, 64), "FVRA": (1442, 826), "FVFA": (1583, 1507)},
                 "cat_prob": {"RVRA": (0.0894, 0.0942), "RVFA": (0.9313, 0.9004), "FVRA": (0.9915, 0.7107), "FVFA": (0.9987, 0.9600)}},
    "NoSSL": {"orig": dict(balanced_accuracy=0.9998, macro_f1=0.9965, mcc=0.9930, roc_auc=1.0, tn=72, fp=0, fn=1, tp=3101),
              "norm": dict(balanced_accuracy=0.8772, macro_f1=0.5095, mcc=0.2552, roc_auc=0.9645, tn=72, fp=0, fn=762, tp=2340),
              "cat_correct": {"RVRA": (72, 72), "RVFA": (74, 73), "FVRA": (1444, 786), "FVFA": (1583, 1481)},
              "cat_prob": {"RVRA": (0.0547, 0.0587), "RVFA": (0.9484, 0.8919), "FVRA": (0.9943, 0.5716), "FVFA": (0.9986, 0.9035)}}}
VERIFIED_CELL53_TESTS = {  # accuracy change (pp), 95% CI, Holm p; Cell 53 output
    "Full_SSL": [(0.0, 0.0, 0.0, 1.0), (-2.083, -6.25, 0.0, 1.0), (-10.417, -20.833, -2.083, 0.3125), (-14.583, -25.0, -6.25, 0.109)],
    "NoSSL": [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (-14.583, -25.0, -6.25, 0.109), (-16.667, -27.083, -6.25, 0.0625)]}
VERIFIED_CELL53_PROBS = """
O:0.060132,0.239160;0.002503,0.001022;0.002915,0.001239;0.312306,0.014230;0.012172,0.005040;0.011378,0.007872;0.006664,0.001886;0.007130,0.002366;0.136798,0.205222;0.076910,0.029556;0.019479,0.003548;0.060884,0.029877;0.994727,0.988672;0.964941,0.997070;0.984180,0.993750;0.990137,0.996484;0.979785,0.950195;0.808472,0.837305;0.994043,0.997656;0.986719,0.997363;0.983887,0.987207;0.962793,0.935547;0.981152,0.964941;0.995898,0.996387;0.994336,0.995117;0.995801,0.996387;0.995313,0.995020;0.995508,0.995605;0.992480,0.995117;0.993164,0.995215;0.996484,0.997070;0.991699,0.995410;0.994434,0.995215;0.992188,0.995410;0.992480,0.996094;0.995215,0.996875;0.997949,0.998242;0.999219,0.999414;0.999609,0.999121;0.999316,0.998730;0.998926,0.998437;0.998047,0.998437;0.999316,0.999609;0.998242,0.998535;0.999512,0.998926;0.999512,0.999316;0.999219,0.999219;0.998730,0.999023
C:0.047375,0.305872;0.003718,0.001607;0.003806,0.006675;0.379390,0.154614;0.184513,0.062721;0.071962,0.074200;0.193521,0.184554;0.084836,0.110681;0.360614,0.266141;0.255231,0.226437;0.124704,0.096409;0.229031,0.195358;0.996777,0.994043;0.992480,0.997949;0.983105,0.991602;0.990820,0.996680;0.990625,0.988867;0.951465,0.963184;0.994434,0.998340;0.990820,0.998340;0.995020,0.995117;0.971973,0.986816;0.990918,0.990039;0.995215,0.983496;0.994434,0.995215;0.996289,0.996680;0.995703,0.995313;0.995410,0.995898;0.993457,0.995508;0.993945,0.995898;0.996680,0.997168;0.992480,0.995801;0.995410,0.996289;0.992285,0.995215;0.993262,0.996289;0.995410,0.996777;0.998047,0.998340;0.999219,0.999414;0.999609,0.999121;0.999414,0.998828;0.999023,0.998535;0.998145,0.998437;0.999316,0.999609;0.998340,0.998633;0.999512,0.998926;0.999512,0.999316;0.999219,0.999219;0.998730,0.999023
A:0.052884,0.272205;0.003728,0.001586;0.003631,0.006761;0.340126,0.158363;0.179781,0.053533;0.068948,0.062053;0.191702,0.182756;0.076826,0.108221;0.392395,0.288007;0.241418,0.226121;0.105127,0.069071;0.229522,0.193897;0.989453,0.993164;0.956348,0.996777;0.978711,0.990625;0.990527,0.995898;0.980566,0.992676;0.735834,0.673828;0.988770,0.996875;0.964551,0.996582;0.994238,0.994727;0.883203,0.882910;0.980273,0.985352;0.995020,0.996191;0.994336,0.995410;0.995996,0.995508;0.996191,0.995996;0.995313,0.995898;0.993359,0.995605;0.993945,0.995801;0.996191,0.997070;0.992480,0.995801;0.995215,0.996289;0.992578,0.995605;0.993066,0.996094;0.995703,0.996875;0.997363,0.998145;0.999414,0.999609;0.999316,0.998926;0.999219,0.998828;0.998730,0.998633;0.997852,0.998340;0.999316,0.999609;0.998535,0.998926;0.999219,0.998730;0.999316,0.999121;0.999219,0.999512;0.997363,0.998535
V:0.052084,0.351184;0.004717,0.002082;0.004604,0.012344;0.376050,0.086413;0.196832,0.070166;0.072511,0.039647;0.197311,0.169746;0.074323,0.040947;0.383105,0.369882;0.287521,0.328223;0.119751,0.089744;0.264696,0.183883;0.995801,0.988867;0.992188,0.997559;0.987988,0.985645;0.988770,0.976953;0.991602,0.957617;0.935449,0.951465;0.993164,0.997363;0.988086,0.995313;0.994727,0.994531;0.972949,0.955762;0.989941,0.984668;0.993359,0.934863;0.238425,0.637054;0.892285,0.133524;0.995508,0.995605;0.993750,0.388517;0.970898,0.836133;0.992285,0.995410;0.521590,0.515988;0.451562,0.049216;0.607486,0.376692;0.379640,0.020232;0.925391,0.229848;0.965039,0.583252;0.978320,0.640363;0.999316,0.985449;0.995215,0.960547;0.996484,0.786316;0.924707,0.713245;0.998535,0.999219;0.991992,0.997070;0.998828,0.999414;0.999512,0.999316;0.999512,0.999805;0.999219,0.999414;0.990430,0.760767
B:0.082266,0.283936;0.003379,0.001406;0.002925,0.001462;0.265025,0.010773;0.048375,0.005103;0.012287,0.020761;0.007903,0.001447;0.007160,0.002596;0.160114,0.180490;0.084479,0.056084;0.018646,0.003639;0.177997,0.023089;0.988672,0.959082;0.939551,0.988477;0.982617,0.979688;0.979492,0.947754;0.944238,0.801099;0.728864,0.624133;0.985254,0.992480;0.972754,0.993750;0.977734,0.968359;0.937109,0.740723;0.961719,0.887305;0.993262,0.992480;0.175854,0.521835;0.891113,0.118163;0.995020,0.994922;0.992285,0.388648;0.912695,0.670874;0.990820,0.994141;0.440673,0.425473;0.279438,0.007046;0.608130,0.397530;0.432881,0.040388;0.736877,0.207073;0.986035,0.699512;0.964453,0.635501;0.999219,0.980664;0.993750,0.978613;0.993848,0.735864;0.891699,0.694727;0.998340,0.999316;0.991113,0.997266;0.998438,0.998535;0.999512,0.999316;0.999414,0.999707;0.999121,0.999023;0.862598,0.686523
"""  # Cell 53 output: per condition, 48 videos x (Full_SSL, NoSSL)
NCAT = {"RVRA": 72, "RVFA": 75, "FVRA": 1444, "FVFA": 1583}

# =============================================================================
# Helpers
# =============================================================================
class Paths:
    def __init__(self, a):
        self.results = Path(a.results); self.project = Path(a.project) if a.project else None
        self.notebook = Path(a.notebook) if a.notebook else None
        self.out = Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        self.tables = self.results / "02_Tables"
        self.pubfigs = self.results / "01_Figures" / "Final_Publication_Figures"
    def proj(self, *p):
        return self.project.joinpath(*p) if self.project else None

def note(msg): print("   note:", msg)

def new_fig(h_mm):
    return plt.figure(figsize=(FW / 25.4, h_mm / 25.4)), h_mm

def axmm(fig, H, x, y, w, h):
    return fig.add_axes([x / FW, y / H, w / FW, h / H])

def letter(fig, H, x, y, s):
    fig.text(x / FW, y / H, s, fontsize=10, fontweight="bold", va="top", ha="left")

def head(fig, H, x, y, s):
    fig.text(x / FW, y / H, s, fontsize=7, va="top")

def save(fig, P, name):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(P.out / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"   saved {name}.pdf/.png/.tiff")

def ygrid(ax): ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
def xgrid(ax): ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)

def find_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} in {list(df.columns)}")

def to_rel(p):
    p = str(p).replace("\\", "/")
    for c, _ in CATS:
        k = p.find(c + "/")
        if k >= 0:
            return p[k:]
    return p

def mcc(tn, fp, fn, tp):
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else 0.0

def bal_acc(tn, fp, fn, tp):
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))

def load_cell56(P):
    """Cell 56 metrics; falls back to the verified manuscript values."""
    base = P.proj("12_Full_Test_Codec_Normalization", "Tables")
    out = json.loads(json.dumps(VERIFIED_CELL56))
    if base is None or not (base / "Cell56_overall_metrics.csv").is_file():
        note("Cell 56 tables not found; using verified manuscript values.")
        return out
    ov = pd.read_csv(base / "Cell56_overall_metrics.csv")
    ov = ov[ov.policy == "Frozen_Validation"]
    for m in ["Full_SSL", "NoSSL"]:
        for enc, key in [("Original", "orig"), ("Normalized_H264_CRF18", "norm")]:
            r = ov[(ov.model == m) & (ov.encoding == enc)].iloc[0]
            out[m][key] = {k: float(r[k]) for k in ["balanced_accuracy", "macro_f1", "mcc", "roc_auc", "tn", "fp", "fn", "tp"]}
    pc = pd.read_csv(base / "Cell56_per_category.csv")
    pc = pc[pc.policy == "Frozen_Validation"]
    for m in ["Full_SSL", "NoSSL"]:
        for _, r in pc[pc.model == m].iterrows():
            out[m]["cat_correct"][r.category] = (int(r.correct_original), int(r.correct_normalized))
            out[m]["cat_prob"][r.category] = (float(r.mean_prob_original), float(r.mean_prob_normalized))
    return out

def _verified_cell53_probs():
    names = {"O": "Original", "C": "Container_Only", "A": "Audio_Only", "V": "Video_Only", "B": "Combined"}
    rows = []
    for line in VERIFIED_CELL53_PROBS.strip().splitlines():
        code, vals = line.split(":")
        for i, pair in enumerate(vals.split(";"), start=1):
            a, b = pair.split(",")
            rows.append((names[code], i, float(a), float(b)))
    return pd.DataFrame(rows, columns=["condition", "idx", "Full_SSL", "NoSSL"])

def _notebook_has_cell53(path):
    try:
        cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    except Exception:
        return False
    for c in cells:
        if c.get("cell_type") == "code" and "CELL 53" in "".join(c.get("source", []))[:300]:
            out = "".join("".join(o.get("text", "")) for o in c.get("outputs", []) if "text" in o)
            return "CONDITION: Combined" in out
    return False

def find_notebook(given, project):
    if given and Path(given).is_file() and _notebook_has_cell53(given):
        return Path(given)
    roots = [r for r in [project, "/content/drive/MyDrive", "/content"] if r and Path(r).is_dir()]
    print("   searching for the development notebook with Cell 53 output ...")
    seen = set()
    for r in roots:
        for nb in sorted(Path(r).glob("**/*.ipynb"), key=lambda p: -p.stat().st_size):
            if nb in seen or ".ipynb_checkpoints" in str(nb):
                continue
            seen.add(nb)
            if _notebook_has_cell53(nb):
                print(f"   found: {nb}")
                return nb
    return None

def load_cell53(P):
    """Per-video probabilities and paired tests printed by Cell 53."""
    probs, tests = None, None
    if P.notebook and P.notebook.is_file():
        cells = json.loads(P.notebook.read_text(encoding="utf-8"))["cells"]
        text = None
        for c in cells:
            if c.get("cell_type") == "code" and "CELL 53" in "".join(c.get("source", []))[:300]:
                text = "".join("".join(o.get("text", "")) for o in c.get("outputs", []) if "text" in o)
        if text:
            rows, cond = [], None
            for line in text.splitlines():
                m = re.match(r"CONDITION: (\S+)", line)
                if m:
                    cond = m.group(1); continue
                m = re.match(r"\[(\d+)/48\] Full SSL=([\d.]+) \| NoSSL=([\d.]+)", line)
                if m and cond:
                    rows.append((cond, int(m.group(1)), float(m.group(2)), float(m.group(3))))
            if len(rows) == 240:
                probs = pd.DataFrame(rows, columns=["condition", "idx", "Full_SSL", "NoSSL"])
            tests = {"Full_SSL": [], "NoSSL": []}
            for cond in ["Container_Only", "Audio_Only", "Video_Only", "Combined"]:
                for m in tests:
                    mm = re.search(rf"{'Full_SSL' if m == 'Full_SSL' else 'NoSSL'}\s+{cond}_vs_Original\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+(\S+)", text)
                    if mm:
                        d, lo, hi, holm = (float(x) for x in mm.groups())
                        tests[m].append((100 * d, 100 * lo, 100 * hi, holm))
            if any(len(v) != 4 for v in tests.values()):
                tests = None
    if probs is None:
        note("Cell 53 output not found in a notebook; using the verified Cell 53 values built into this script.")
        probs = _verified_cell53_probs()
    if tests is None:
        note("Cell 53 paired tests not parsed; using verified manuscript values.")
        tests = VERIFIED_CELL53_TESTS
    return probs, tests

# =============================================================================
# Figure 2 — sample audiovisual inputs
# =============================================================================
def logmel(wav, sr=16000, n_fft=640, win=400, hop=160, n_mels=128, fmax=8000):
    """Log-mel spectrogram with numpy only (HTK mel scale), in dB relative to max."""
    if len(wav) < win:
        wav = np.pad(wav, (0, win - len(wav)))
    frames = 1 + (len(wav) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(wav[idx] * np.hanning(win), n=n_fft)) ** 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l: fb[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    mel = spec @ fb.T
    db = 10 * np.log10(np.maximum(mel, 1e-10))
    return np.maximum(db - db.max(), -80).T          # (n_mels, frames), 0 … −80 dB

def fig2_samples(P):
    import cv2
    root = P.proj("08_Representative_Dataset_Samples")
    vids = {}
    if root and root.is_dir():
        for folder, code in CATS:
            hits = sorted(p for p in root.rglob("*.mp4") if folder in str(p))
            if hits: vids[code] = hits[0]
    if len(vids) < 4:
        src = P.pubfigs / "Dataset_Visualization" / "Figure_40A_Refined_Actual_FakeAVCeleb_AV_Samples.png"
        note("sample videos not found; copying the published sample figure from the results package.")
        im = Image.open(src).convert("RGB")
        for ext in ["png", "tiff", "pdf"]:
            im.save(P.out / f"Figure2_Sample_Inputs.{ext}", dpi=(600, 600))
        return
    fig, H = new_fig(134)
    labels = {"RVRA": ("Real video · real audio", REAL), "RVFA": ("Real video · fake audio", "#C98F3A"),
              "FVRA": ("Fake video · real audio", "#2C6BAC"), "FVFA": ("Fake video · fake audio", "#8E3F7A")}
    for r, (_, code) in enumerate(CATS):
        path = vids[code]
        cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        y = H - 12 - r * 30
        for j, frac in enumerate([0.2, 0.5, 0.8]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * max(n - 1, 0)))
            ok, fr = cap.read()
            ax = axmm(fig, H, 20 + j * 25, y - 22, 22, 22); ax.axis("off")
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); h, w = fr.shape[:2]; s = min(h, w)
                ax.imshow(fr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2])
            ax.set_title(f"{frac * n / fps:.2f} s", fontsize=5.4, color=MUTED, pad=1.5, y=-0.16)
        cap.release()
        with tempfile.TemporaryDirectory() as td:
            wavp = Path(td) / "a.f32"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "16000",
                            "-f", "f32le", str(wavp)], check=True)
            wav = np.fromfile(wavp, dtype=np.float32)
        S = logmel(wav)
        ax = axmm(fig, H, 104, y - 22, 58, 22)
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                       extent=(0, len(wav) / 16000, 0, S.shape[0]))
        ax.set_ylabel("Mel bin", fontsize=5.6); ax.tick_params(labelsize=5.2)
        if r == 3: ax.set_xlabel("Time (s)", fontsize=5.6)
        fig.text(3 / FW, (y - 9) / H, code, fontsize=7.5, fontweight="bold", color=labels[code][1], rotation=90, va="center")
        fig.text(20 / FW, (y + 2.5) / H, f"({'abcd'[r]})  {labels[code][0]}", fontsize=6.2, va="bottom")
    cax = axmm(fig, H, 165, 14, 2.2, H - 30)
    cb = fig.colorbar(im, cax=cax, ticks=[-80, -60, -40, -20, 0])
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    cb.set_label("Relative power (dB)", fontsize=5.6); cb.ax.tick_params(labelsize=5.2)
    save(fig, P, "Figure2_Sample_Inputs")

# =============================================================================
# Figure 3 — dataset distribution
# =============================================================================
CURATION = {  # Cells 4, 6B, 7, 9 and 11 of the development notebook
    "downloaded": 21560, "metadata_rows": 21566,
    "conflict_groups": 25, "conflict_files": 51, "after_conflicts": 21509,
    "same_label_dups": 17, "after_dedup": 21492,
    "no_metadata": 16, "final": 21476, "undecodable": 0,
    "authentic_before": 500, "authentic_final": 475}

def _box(ax, x, y, w, h, lines, fc, ec, bold_first=True, fs=5.6):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.7, zorder=2))
    n = len(lines); lh = fs * 0.43
    for k, t in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1) * lh / 2 - k * lh, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if (bold_first and k == 0) else "normal", color=INK, zorder=3)

def _arrow(ax, p, q, c=INK):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=c, lw=0.7, shrinkA=0, shrinkB=0, zorder=1))

def fig3_distribution(P):
    d = P.pubfigs / "Dataset_Visualization"
    split = pd.read_csv(d / "Dataset_Split_Counts.csv")
    cat = pd.read_csv(d / "Dataset_Category_By_Split.csv").set_index("split")
    binc = pd.read_csv(d / "Dataset_Binary_Class_By_Split.csv").set_index("split")
    C = CURATION
    assert int(binc.sum().sum()) == C["final"], "split files do not add up to the curated cohort"
    fig, H = new_fig(150)

    # ---------------- (a) cohort curation flow chart
    ax = fig.add_axes([0, 64 / H, 1, 80 / H]); ax.set_xlim(0, 180); ax.set_ylim(0, 80); ax.axis("off")
    MAIN, MAIN_E = "#EEF3F9", "#2C6BAC"; EXC, EXC_E = "#FBEDED", "#B23A3A"; FIN, FIN_E = "#E7F2EA", REAL
    w, h, y = 31, 17, 57
    xs = [2, 37.5, 73, 108.5, 144]
    main = [["Files downloaded", f"{C['downloaded']:,} videos", f"({C['metadata_rows']:,} metadata rows)"],
            ["Label conflicts removed", f"{C['after_conflicts']:,} videos"],
            ["Exact duplicates removed", f"{C['after_dedup']:,} videos"],
            ["Metadata matched", f"{C['final']:,} videos"],
            ["Final clean cohort", f"{C['final']:,} videos", f"{C['authentic_final']} authentic", f"{C['final'] - C['authentic_final']:,} manipulated", "500 source identities"]]
    for k, (x, lines) in enumerate(zip(xs, main)):
        last = k == len(xs) - 1
        _box(ax, x, y, w, h, lines, FIN if last else MAIN, FIN_E if last else MAIN_E)
        if k: _arrow(ax, (xs[k - 1] + w, y + h / 2), (x, y + h / 2))
    exc = [[f"−{C['conflict_files']} files", f"{C['conflict_groups']} byte-identical groups", "labelled both authentic", "and manipulated"],
           [f"−{C['same_label_dups']} files", "byte-identical copies", "with the same label"],
           [f"−{C['no_metadata']} files", "no metadata record"],
           [f"−{C['undecodable']} files", "decodability and", "audio–video integrity audit"]]
    for k, lines in enumerate(exc):
        xm = (xs[k] + w + xs[k + 1]) / 2
        bx = xm - 15.5
        _box(ax, bx, 34, 31, 14, lines, EXC, EXC_E, fs=5.3)
        ax.plot([xm, xm], [y + h / 2, 48], color=EXC_E, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    # split
    fx = xs[-1] + w / 2
    ax.plot([fx, fx], [y, 28], color=INK, lw=0.7)
    ax.text(90, 29.4, "Source-identity-disjoint split (seed 42)", ha="center", fontsize=5.8, fontweight="bold")
    ax.text(90, 26.0, "0 shared source identities · 0 shared SHA-256 hashes between partitions", ha="center", fontsize=5.2, color=MUTED)
    ax.plot([24, fx], [24, 24], color=INK, lw=0.7)
    sx = [8, 66, 124]; sw = 48
    for k, part in enumerate(["Train", "Validation", "Test"]):
        a_, m_ = int(binc.loc[part, "Authentic"]), int(binc.loc[part, "Manipulated"])
        ids = {"Train": 350, "Validation": 75, "Test": 75}[part]
        _arrow(ax, (sx[k] + sw / 2, 24), (sx[k] + sw / 2, 17))
        _box(ax, sx[k], 2, sw, 15, [f"{part}: {a_ + m_:,} videos", f"{ids} source identities",
                                     f"{a_} authentic · {m_:,} manipulated"], MAIN, MAIN_E, fs=5.6)
    ax.plot([sx[0] + sw / 2, sx[2] + sw / 2], [24, 24], color=INK, lw=0.7)
    letter(fig, H, 2, 148, "a"); head(fig, H, 8, 147.6, "Cohort curation and split")
    fig.text(176 / FW, 147.6 / H, f"authentic videos: {C['authentic_before']} → {C['authentic_final']}", ha="right", va="top",
             fontsize=5.4, color=MUTED)

    # ---------------- (b–d) distributions
    panels = [("Videos per partition", split.Partition.tolist(), split.Videos.tolist(), ["#2C6BAC", "#3E9C8F", "#D27A2C"]),
              ("Test videos per category", [c for _, c in CATS], [int(cat.loc["Test", c]) for _, c in CATS],
               [REAL, "#C98F3A", "#2C6BAC", "#8E3F7A"]),
              ("Test videos per class", ["Authentic", "Manipulated"],
               [int(binc.loc["Test", "Authentic"]), int(binc.loc["Test", "Manipulated"])], [REAL, FAKE])]
    for k, (title, labs, vals, cols) in enumerate(panels):
        ax = axmm(fig, H, 14 + k * 57, 9, 44, 40)
        ax.bar(range(len(vals)), vals, color=cols, width=0.68)
        for i_, v in enumerate(vals):
            ax.text(i_, v * 1.01 + max(vals) * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=5.6)
        ax.set_xticks(range(len(vals)), labs); ax.set_ylim(0, max(vals) * 1.15)
        ax.set_title(title, pad=4); ygrid(ax)
        if k == 0: ax.set_ylabel("Number of videos")
        letter(fig, H, 2 + k * 57, 59, "bcd"[k])
    tot = int(binc.loc["Test"].sum()); share = binc.loc["Test", "Manipulated"] / tot
    fig.axes[-1].text(1, binc.loc["Test", "Manipulated"] * 0.5, f"{100 * share:.1f}%", ha="center", color="white",
                      fontsize=7, fontweight="bold")
    save(fig, P, "Figure3_Dataset_Curation_and_Distribution")

# =============================================================================
# Figure 4 — training progress
# =============================================================================
def fig4_training(P):
    T = P.tables
    fig, H = new_fig(70)
    ax = axmm(fig, H, 16, 13, 80, 47)
    for m, f in [("Full_SSL", "smdnet_v2_full_training_history_fast.csv"),
                 ("NoSSL", "smdnet_v2_nossl_training_history_fast.csv"),
                 ("Visual_Only", "smdnet_v2_visual_only_training_history_fast.csv")]:
        h = pd.read_csv(T / f)
        ax.plot(h.epoch, h.fast_val_roc_auc, color=COL[m], lw=1.0, marker="o", ms=1.8, label=NAME[m])
        b = h.loc[h.fast_val_roc_auc.idxmax()]
        ax.plot(b.epoch, b.fast_val_roc_auc, marker="*", ms=7, color=COL[m], mec="white", mew=0.5, zorder=5)
    hdl, lab = ax.get_legend_handles_labels()
    hdl.append(Line2D([], [], ls="none", marker="*", ms=7, color=MUTED, mec="white")); lab.append("selected checkpoint")
    ax.legend(hdl, lab, frameon=False, loc="lower right")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation ROC-AUC (fast subset)"); ax.set_ylim(0.70, 1.01); ax.set_xlim(0, 31)
    ax.set_title("Stage B supervised training", pad=4); ygrid(ax); letter(fig, H, 2, 68, "a")

    s = pd.read_csv(T / "stageA_ssl_training_history.csv")
    ax = axmm(fig, H, 118, 13, 46, 47)
    ax.plot(s.epoch, s.train_loss, color="#9A86CF", lw=1.0, marker="o", ms=1.8, label="training")
    ax.plot(s.epoch, s.val_loss, color=COL["Full_SSL"], lw=1.0, marker="o", ms=1.8, label="validation")
    b = s.loc[s.val_loss.idxmin()]
    ax.plot(b.epoch, b.val_loss, marker="*", ms=7, color=COL["Full_SSL"], mec="white", mew=0.5, zorder=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("InfoNCE loss"); ax.set_xticks(range(2, int(s.epoch.max()) + 1, 2))
    ax.set_ylim(2.0, 4.9); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.18)); ygrid(ax)
    ax2 = ax.twinx()
    ax2.bar(s.epoch, 100 * s.val_mean_top1, width=0.5, color="#D9D9D9", zorder=0)
    ax2.axhline(100 / 72, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax2.text(0.7, 100 / 72 + 0.45, "chance (1/72)", fontsize=5.2, color=MUTED)
    ax2.set_ylim(0, 20); ax2.set_ylabel("Retrieval top-1 (%)", color=MUTED); ax2.tick_params(axis="y", colors=MUTED)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title("Stage A pretraining", pad=4); letter(fig, H, 106, 68, "b")
    save(fig, P, "Figure4_Training_Progress")

# =============================================================================
# Figure 5 — ROC curves
# =============================================================================
def roc_points(y, p):
    order = np.argsort(-p, kind="mergesort"); y = y[order]; p = p[order]
    distinct = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tps = np.cumsum(y)[distinct]; fps = np.cumsum(1 - y)[distinct]
    tpr = np.r_[0, tps / y.sum()]; fpr = np.r_[0, fps / (1 - y).sum()]
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # NumPy 2 renamed trapz
    return fpr, tpr, trap(tpr, fpr)

def fig5_roc(P):
    pred_dir = P.proj("05_Predictions")
    files = {"Full_SSL": ("smdnet_v2_final_test_5window_predictions.csv", "probability_manipulated"),
             "NoSSL": ("smdnet_v2_nossl_test_5window_predictions.csv", "nossl_probability"),
             "Visual_Only": ("smdnet_v2_visual_only_test_5window_predictions.csv", "visual_only_probability")}
    if pred_dir is None or not all((pred_dir / f).is_file() for f, _ in files.values()):
        note("prediction files not found in <project>/05_Predictions; copying the published ROC figure.")
        im = Image.open(P.pubfigs / "Figure_38A_ROC_Curves.png").convert("RGB")
        for ext in ["png", "tiff", "pdf"]:
            im.save(P.out / f"Figure5_ROC_Curves.{ext}", dpi=(600, 600))
        return
    fig, H = new_fig(84)
    ax = axmm(fig, H, 16, 12, 66, 66); axz = axmm(fig, H, 104, 12, 66, 66)
    for m, (f, col) in files.items():
        d = pd.read_csv(pred_dir / f)
        rel = d[find_col(d, ["relative_path", "video_path", "path"])].map(to_rel)
        y = (~rel.str.startswith("RealVideo-RealAudio/")).astype(int).values
        fpr, tpr, auc = roc_points(y, d[col].astype(float).values)
        for a in (ax, axz):
            a.plot(fpr, tpr, color=COL[m], lw=1.2, label=f"{NAME[m]} (AUC = {auc:.4f})", drawstyle="steps-post")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="chance")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_title("Full range", pad=4)
    axz.set_xlim(-0.005, 0.10); axz.set_ylim(0.90, 1.002); axz.set_title("Zoom: FPR ≤ 0.10, TPR ≥ 0.90", pad=4)
    for a in (ax, axz):
        a.set_xlabel("False positive rate"); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    ax.set_ylabel("True positive rate"); ax.legend(frameon=False, loc="lower right")
    letter(fig, H, 2, 82, "a"); letter(fig, H, 92, 82, "b")
    save(fig, P, "Figure5_ROC_Curves")

# =============================================================================
# Figure 6 — confusion matrices
# =============================================================================
def fig6_confusion(P):
    perf = pd.read_csv(P.tables / "Final_Publication_Tables" / "Table_02_Frozen_Validation_Threshold_Performance.csv")
    orig = {}
    for m in ["Full_SSL", "NoSSL", "Visual_Only"]:
        r = perf[perf.Model == m].iloc[0]
        orig[m] = (int(r.TN), int(r.FP), int(r.FN), int(r.TP))
        if orig[m] != VERIFIED_CONFUSION_ORIGINAL[m]:
            warnings.warn(f"{m}: confusion counts differ from the manuscript values")
    c56 = load_cell56(P)
    norm = {m: tuple(int(c56[m]["norm"][k]) for k in ["tn", "fp", "fn", "tp"]) for m in ["Full_SSL", "NoSSL"]}
    GREEN, RED = np.array([46, 125, 79]) / 255, np.array([178, 58, 58]) / 255
    tint = lambda base, f: tuple(1 - (1 - base) * (0.12 + 0.88 * f))

    def matrix(ax, m, cm):
        tn, fp, fn, tp = cm; cells = [[tn, fp], [fn, tp]]; tot = [tn + fp, fn + tp]
        for r in range(2):
            for c in range(2):
                v = cells[r][c]; pct = v / tot[r]; p = 100 * pct
                ax.add_patch(Rectangle((c, 1 - r), 1, 1, fc=tint(GREEN if r == c else RED, pct if v else 0), ec="white", lw=2))
                dark = pct > 0.55
                ax.text(c + .5, 1 - r + .58, f"{v:,}", ha="center", va="center", fontsize=9, fontweight="bold", color="white" if dark else INK)
                ptxt = f"{p:.2f}%" if (0 < p < 0.1 or 99.9 < p < 100) else f"{p:.1f}%"
                ax.text(c + .5, 1 - r + .30, ptxt, ha="center", va="center", fontsize=6, color="white" if dark else MUTED)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
        ax.set_xticks([.5, 1.5], ["Authentic", "Manipulated"])
        ax.set_yticks([1.5, .5], [f"Authentic\n(n = {tn + fp:,})", f"Manipulated\n(n = {fn + tp:,})"])
        ax.tick_params(length=0, labelsize=6)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlabel("Predicted label", fontsize=6, color=MUTED, labelpad=2)
        ax.set_title(NAME[m], fontsize=7.2, fontweight="bold", color=COL[m], pad=12)
        ax.text(1, 2.08, f"BA {bal_acc(*cm):.4f} · MCC {mcc(*cm):.3f}", ha="center", va="bottom", fontsize=5.6, color=MUTED)

    fig, H = new_fig(132)
    for i, m in enumerate(["Full_SSL", "NoSSL", "Visual_Only"]):
        ax = axmm(fig, H, 24 + i * 58, 76, 36, 36); matrix(ax, m, orig[m])
        if i == 0: ax.set_ylabel("True label", fontsize=6, color=MUTED, labelpad=2)
    for i, m in enumerate(["Full_SSL", "NoSSL"]):
        ax = axmm(fig, H, 53 + i * 58, 12, 36, 36); matrix(ax, m, norm[m])
        if i == 0: ax.set_ylabel("True label", fontsize=6, color=MUTED, labelpad=2)
    letter(fig, H, 2, 129, "a"); head(fig, H, 8, 128.6, "Original test set (frozen validation thresholds)")
    letter(fig, H, 2, 65, "b"); head(fig, H, 8, 64.6, "Same test videos re-encoded to H.264 (CRF 18) with 16 kHz AAC: manipulations unchanged")
    for i, (lab, base) in enumerate([("correct (shade = row %)", GREEN), ("error (shade = row %)", RED)]):
        fig.patches.append(Rectangle((6 / FW, (36 - 6 * i) / H), 4 / FW, 4 / H, transform=fig.transFigure, fc=tint(base, .8), ec="none"))
        fig.text(12 / FW, (38 - 6 * i) / H, lab, fontsize=5.8, va="center")
    fig.text(6 / FW, 23 / H, "Each cell: count and\npercentage of its true class.", fontsize=5.6, va="center", color=MUTED)
    save(fig, P, "Figure6_Confusion_Matrices")

# =============================================================================
# Figure 7 — ablation and error analysis
# =============================================================================
def fig7_ablation(P):
    T = P.tables
    ci = pd.read_csv(T / "Final_Publication_Tables" / "Cell37_Statistical_Validation" / "Table_37C_Model_95CI.csv")
    fig, H = new_fig(118)
    for k, (metric, lab, lim) in enumerate([("balanced_accuracy", "Balanced accuracy", (0.984, 1.0015)),
                                            ("macro_f1", "Macro-F1", (0.80, 1.005))]):
        ax = axmm(fig, H, 15 + 37 * k, 66, 25, 40)
        for i, m in enumerate(["Full_SSL", "NoSSL", "Visual_Only"]):
            r = ci[(ci.model == m) & (ci.metric == metric)].iloc[0]
            ax.errorbar(i, r.point_estimate, yerr=[[r.point_estimate - r.ci_95_lower], [r.ci_95_upper - r.point_estimate]],
                        fmt="o", ms=3.6, color=COL[m], elinewidth=0.9, capsize=2.2, capthick=0.8)
        ax.set_xticks(range(3), ["Full\nSSL", "No\nSSL", "Visual\nonly"]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(*lim)
        ax.set_title(lab, pad=4); ygrid(ax)
    fig.text(46 / FW, 54.5 / H, "Majority-class baseline (always \u201cmanipulated\u201d): balanced accuracy 0.500,\n"
             "macro-F1 0.494 \u2014 both off scale. Error bars: 95% stratified bootstrap CIs.",
             ha="center", va="center", fontsize=5.3, color=MUTED, style="italic")
    letter(fig, H, 2, 114, "a")

    pc = pd.read_csv(T / "smdnet_v2_three_model_per_condition.csv")
    pc = pc[pc.operating_point == "Validation_Frozen"]
    mods = [("Full_SSL", "Full SSL"), ("Full_NoSSL", "No SSL"), ("Visual_Only", "Visual only")]
    M = np.zeros((3, 4)); TX = [[""] * 4 for _ in range(3)]
    for i, (m, _) in enumerate(mods):
        for j, (c, _) in enumerate(CATS):
            r = pc[(pc.model == m) & (pc.condition == c)].iloc[0]
            M[i, j] = r.accuracy; TX[i][j] = f"{int(r.correct):,}/{int(r.n):,}"
    ax = axmm(fig, H, 104, 66, 72, 40)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("acc", [(0, "#B23A3A"), (0.55, "#EDBFAE"), (0.88, "#F8EEE8"), (1, "#E8F2EA")])
    ax.imshow(M, cmap=cmap, vmin=0.25, vmax=1.0, aspect="auto")
    for i in range(3):
        for j in range(4):
            dark = M[i, j] < 0.5
            ax.text(j, i - .12, f"{100 * M[i, j]:.1f}%", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white" if dark else INK)
            ax.text(j, i + .22, TX[i][j], ha="center", va="center", fontsize=5.0, color="white" if dark else MUTED)
    ax.set_xticks(range(4), [c for _, c in CATS]); ax.set_yticks(range(3), [m[1] for m in mods]); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    ax.set_title("Accuracy by category (frozen thresholds)", pad=4); letter(fig, H, 88, 114, "b")

    d = pd.read_csv(T / "smdnet_v2_three_model_error_overlap.csv")
    a, b, c = (d[k].astype(bool) for k in ["error_frozen_Full_SSL", "error_frozen_Full_NoSSL", "error_frozen_Visual_Only"])
    groups = [("Visual only", ~a & ~b & c), ("Full SSL", a & ~b & ~c), ("No SSL", ~a & b & ~c),
              ("Full SSL +\nVisual only", a & ~b & c), ("Full SSL +\nNo SSL", a & b & ~c),
              ("No SSL +\nVisual only", ~a & b & c), ("all three", a & b & c)]
    groups = [(n, int(g.sum())) for n, g in groups if g.sum() > 0]
    ax = axmm(fig, H, 40, 11, 110, 30)
    y = np.arange(len(groups))[::-1]
    ax.barh(y, [g[1] for g in groups], color="#8C8C8C", height=0.58)
    for yi, (_, v) in zip(y, groups): ax.text(v + 1, yi, str(v), va="center", fontsize=6)
    ax.set_yticks(y, [g[0] for g in groups]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(g[1] for g in groups) * 1.15); ax.set_xlabel("Misclassified test videos")
    ax.set_title("Error overlap", pad=4); letter(fig, H, 2, 47, "c")
    save(fig, P, "Figure7_Ablation_and_Error_Analysis")

# =============================================================================
# Figure 8 — encoding shortcut and codec control
# =============================================================================
def codec_rule(meta):
    v = meta[meta.feature == "video_codec"]
    h264 = v[v.value.astype(str) == "h264"].iloc[0]
    other = v[v.value.astype(str) != "h264"][["authentic", "manipulated"]].sum()
    tn, fn = int(h264.authentic), int(h264.manipulated)          # predicted authentic
    fp, tp = int(other.authentic), int(other.manipulated)        # predicted manipulated
    return tn, fp, fn, tp

def fig8_codec(P):
    T = P.tables
    meta = pd.read_csv(T / "smdnet_v2_metadata_shortcut_indicators.csv")
    perf = pd.read_csv(T / "Final_Publication_Tables" / "Table_02_Frozen_Validation_Threshold_Performance.csv").set_index("Model")
    probs, tests = load_cell53(P)
    fig, H = new_fig(194)

    # (a) encoding attributes vs label
    rows = [("video_codec", "h264", "H.264"), ("video_codec", "mpeg4", "MPEG-4 Part 2"),
            ("audio_sample_rate", "44100", "44.1 kHz"), ("audio_sample_rate", "16000", "16 kHz"),
            ("audio_codec", "mp3", "MP3"), ("audio_codec", "aac", "AAC")]
    groups = {"video_codec": "Video codec", "audio_sample_rate": "Audio sample rate", "audio_codec": "Audio codec"}
    ax = axmm(fig, H, 30, 150, 52, 33); ypos = []
    for k, (feat, val, lab) in enumerate(rows):
        r = meta[(meta.feature == feat) & (meta.value.astype(str) == val)].iloc[0]
        fa = r.authentic / r.total; yy = -(k + (k // 2) * 0.7); ypos.append(yy)
        ax.barh(yy, fa, color=REAL, height=0.72); ax.barh(yy, 1 - fa, left=fa, color=FAKE, height=0.72, alpha=0.85)
        ax.text(1.02, yy, f"{r.authentic:,} / {r.manipulated:,}", va="center", fontsize=5.6)
        if k % 2 == 0:
            ax.text(-0.02, yy + 0.85, groups[feat], ha="right", va="center", fontsize=5.8, fontweight="bold",
                    transform=ax.get_yaxis_transform())
    ax.set_yticks(ypos, [r[2] for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1], ["0", "50", "100"]); ax.set_xlabel("Share of test videos (%)")
    ax.set_ylim(min(ypos) - .7, 1.2); ax.spines["left"].set_visible(False)
    ax.text(1.02, 1.0, "auth. / manip.", fontsize=5.4, color=MUTED, va="center")
    ax.legend([Line2D([], [], lw=5, color=REAL), Line2D([], [], lw=5, color=FAKE, alpha=.85)], ["authentic", "manipulated"],
              frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.2)
    letter(fig, H, 2, 193, "a"); head(fig, H, 9, 192.6, "Encoding attributes are confounded with the label")

    # (b) codec rule vs trained models
    rule = codec_rule(meta)
    entries = [("Majority class", 0.5, MUTED), ("Codec rule\n(no training)", bal_acc(*rule), FAKE)] + \
              [(NAME[m], float(perf.loc[m, "Balanced_Accuracy"]), COL[m]) for m in ["Visual_Only", "Full_SSL", "NoSSL"]]
    ax = axmm(fig, H, 128, 150, 46, 33)
    for i, (lab, v, c) in enumerate(entries):
        ax.plot([0.45, v], [-i, -i], color=c, lw=1.0, alpha=.6); ax.plot(v, -i, "o", ms=4.2, color=c, mec="white", mew=.5)
        ax.text(v - .012, -i + .33, f"{v:.3f}", ha="right", fontsize=5.6, color=c, fontweight="bold" if "Codec" in lab else "normal")
    ax.set_yticks([-i for i in range(len(entries))], [e[0] for e in entries]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.45, 1.03); ax.set_ylim(-len(entries) + .4, .6); ax.set_xlabel("Balanced accuracy on the test set")
    ax.spines["left"].set_visible(False); xgrid(ax)
    letter(fig, H, 98, 193, "b"); head(fig, H, 105, 192.6, "A one-line codec rule nearly matches trained models")

    # (c) per-video paths, 48-video cohort, shown for the two informative categories
    #     (videos 1–12 RVRA, 13–24 RVFA, 25–36 FVRA, 37–48 FVFA)
    order = ["Original", "Container_Only", "Audio_Only", "Video_Only", "Combined"]
    xlab = ["orig.", "remux", "audio\nre-enc.", "video\nre-enc.", "both\nre-enc."]
    panels = [("Full_SSL", "RVRA", range(1, 13), REAL, "authentic (RVRA)"),
              ("Full_SSL", "FVRA", range(25, 37), FAKE, "face swap / lip sync (FVRA)"),
              ("NoSSL", "RVRA", range(1, 13), REAL, "authentic (RVRA)"),
              ("NoSSL", "FVRA", range(25, 37), FAKE, "face swap / lip sync (FVRA)")]
    for k, (m, cat, ids, c, sub) in enumerate(panels):
        x0 = 16 + k * 41 + (4 if k >= 2 else 0)
        ax = axmm(fig, H, x0, 86, 34, 33)
        w = probs[probs.idx.isin(list(ids))].pivot(index="idx", columns="condition", values=m)[order]
        for _, row in w.iterrows():
            ax.plot(range(5), row.values, color=c, lw=0.6, alpha=0.35, zorder=2)
        ax.plot(range(5), w.median().values, color=c, lw=2.0, marker="o", ms=3, zorder=4)
        ax.axhline(THR[m], color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
        correct = (w >= THR[m]).sum() if cat != "RVRA" else (w < THR[m]).sum()
        for xi, n in enumerate(correct.values):
            ax.text(xi, 1.07, f"{n}/12", ha="center", fontsize=5.3,
                    color=FAKE if n < 12 else MUTED, fontweight="bold" if n < 12 else "normal")
        ax.set_xticks(range(5), xlab, fontsize=5.4); ax.set_xlim(-.4, 4.4); ax.set_ylim(-.03, 1.13)
        ax.set_yticks([0, .25, .5, .75, 1.0]); ygrid(ax)
        ax.set_title(sub, fontsize=6.2, color=c, pad=9)
        if k in (0, 2):
            ax.set_ylabel("p(manipulated)")
            ax.text(-0.35, THR[m] + 0.03, f"threshold {THR[m]:.2f}", fontsize=5, color=INK)
        else:
            ax.set_yticklabels([])
    fig.text((16 + 75 / 2) / FW, 129 / H, "Full SSL", ha="center", fontsize=7, fontweight="bold", color=COL["Full_SSL"])
    fig.text((20 + 82 + 75 / 2) / FW, 129 / H, "No SSL", ha="center", fontsize=7, fontweight="bold", color=COL["NoSSL"])
    fig.text(176 / FW, 136.6 / H, "thin lines: 12 videos · bold: median · n/12: correct decisions",
             ha="right", va="top", fontsize=5.3, color=MUTED)
    letter(fig, H, 2, 137, "c")
    head(fig, H, 9, 136.6, "48-video cohort: the same videos, only the encoding changed")

    # (d) accuracy change with CI
    ax = axmm(fig, H, 16, 14, 46, 42)
    for k, m in enumerate(["Full_SSL", "NoSSL"]):
        off = -.13 if k == 0 else .13
        for i, (dlt, lo, hi, _) in enumerate(tests[m]):
            ax.errorbar(i + off, dlt, yerr=[[dlt - lo], [hi - dlt]], fmt="o", ms=3.4, color=COL[m], elinewidth=.9,
                        capsize=2, capthick=.8, label=NAME[m] if i == 0 else None)
    ax.axhline(0, color=INK, lw=.6); ax.set_xticks(range(4), ["remux", "audio\nre-enc.", "video\nre-enc.", "both\nre-enc."])
    ax.set_xlim(-.5, 3.5); ax.set_ylim(-30, 4); ax.set_ylabel("Accuracy change (pp)"); ygrid(ax)
    ax.legend(frameon=False, loc="lower left", handlelength=1.0)
    letter(fig, H, 2, 68, "d"); head(fig, H, 9, 67.6, "Paired change, 95% CI")

    # (e) Visual_Only CRF sweep: median and interquartile range per group
    pr = pd.read_csv(T / "smdnet_v2_multilevel_compression_predictions.csv")
    lev = ["Original", "CRF10", "CRF18", "CRF23", "CRF30"]
    grp = [("Authentic", "authentic", REAL), ("AudioOnlyFake", "audio-only fake", "#C98F3A"), ("VideoManipulated", "video-manipulated", FAKE)]
    ax = axmm(fig, H, 76, 14, 48, 42)
    for g, lab, c in grp:
        q = np.array([np.percentile(pr[(pr.group == g) & (pr.compression == L)].probability.values, [25, 50, 75]) for L in lev])
        ax.fill_between(range(5), q[:, 0], q[:, 2], color=c, alpha=0.18, lw=0)
        ax.plot(range(5), q[:, 1], color=c, lw=1.6, marker="o", ms=3, label=lab)
    ax.axhline(.5, color=INK, lw=.6, ls=(0, (3, 2))); ax.text(4.35, .52, "0.5", fontsize=5.2, ha="right")
    ax.set_xticks(range(5), ["orig.", "10", "18", "23", "30"]); ax.set_xlabel("H.264 CRF")
    ax.set_xlim(-.2, 4.2); ax.set_ylim(-.02, 1.03); ax.set_ylabel("Visual-only p(manipulated)"); ygrid(ax)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, .02), handlelength=1.4, fontsize=5.4)
    letter(fig, H, 66, 68, "e"); head(fig, H, 73, 67.6, "Visual only: median and IQR")

    # (f) ROC-AUC and authentic accuracy vs CRF
    m35 = pd.read_csv(T / "smdnet_v2_cell35_recovered_robustness_metrics.csv")
    ov = m35[(m35.group == "Overall") & (m35.threshold_policy == "Default_0.5")].set_index("compression").loc[lev]
    au = m35[(m35.group == "Authentic") & (m35.threshold_policy == "Default_0.5")].set_index("compression").loc[lev]
    ax = axmm(fig, H, 138, 14, 38, 42)
    ax.plot(range(5), ov.roc_auc, color=COL["Visual_Only"], marker="o", ms=3, lw=1.3, label="ROC-AUC")
    ax.plot(range(5), au.accuracy, color=REAL, marker="s", ms=2.8, lw=1.3, label="authentic\ncorrect (0.5)")
    ax.axhline(.5, color=MUTED, lw=.6, ls=(0, (3, 2))); ax.text(4.1, .53, "chance", fontsize=5.2, color=MUTED, ha="right")
    ax.set_xticks(range(5), ["orig.", "10", "18", "23", "30"]); ax.set_xlabel("H.264 CRF"); ax.set_ylim(-.03, 1.05); ygrid(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, .04), handlelength=1.4, fontsize=5.4)
    letter(fig, H, 128, 68, "f"); head(fig, H, 135, 67.6, "Visual only: AUC vs CRF")
    save(fig, P, "Figure8_Encoding_Shortcut_and_Codec_Control")

# =============================================================================
# Figure 9 — full-test-set codec normalisation
# =============================================================================
def fig9_fulltest(P):
    c56 = load_cell56(P)
    cats = [c for _, c in CATS]
    acc = {m: {c: (c56[m]["cat_correct"][c][0] / NCAT[c], c56[m]["cat_correct"][c][1] / NCAT[c]) for c in cats} for m in ["Full_SSL", "NoSSL"]}
    prob = {m: {c: tuple(c56[m]["cat_prob"][c]) for c in cats} for m in ["Full_SSL", "NoSSL"]}
    fig, H = new_fig(78)

    def dumbbell(ax, data, fmt, xlab, xlim, thr=False):
        for ci, c in enumerate(cats):
            for k, m in enumerate(["Full_SSL", "NoSSL"]):
                y = -(ci * 2.6 + k); a, b = data[m][c]
                ax.plot([a, b], [y, y], color=COL[m], lw=1.0, alpha=.55, zorder=2)
                ax.plot(a, y, "o", ms=3.6, mfc="white", mec=COL[m], mew=.9, zorder=3)
                ax.plot(b, y, "o", ms=3.6, color=COL[m], zorder=3)
                if abs(a - b) > .08:
                    ax.text(min(a, b) - .015 * (xlim[1] - xlim[0]), y, fmt(b), ha="right", va="center", fontsize=5.4, color=COL[m])
        ax.set_yticks([-(ci * 2.6 + .5) for ci in range(4)], [f"{c}\nn = {NCAT[c]:,}" for c in cats])
        ax.tick_params(axis="y", length=0); ax.set_xlim(*xlim); ax.set_ylim(-(3 * 2.6 + 1) - .7, .8)
        ax.set_xlabel(xlab); xgrid(ax); ax.spines["left"].set_visible(False)
        if thr:
            for m in ["Full_SSL", "NoSSL"]:
                ax.axvline(THR[m], color=COL[m], lw=.6, ls=(0, (3, 2)), zorder=1)

    ax = axmm(fig, H, 18, 13, 40, 52)
    dumbbell(ax, acc, lambda v: f"{100 * v:.0f}%", "Accuracy (frozen threshold)", (.4, 1.03))
    ax.set_xticks([.4, .6, .8, 1.0], ["40%", "60%", "80%", "100%"])
    letter(fig, H, 2, 75, "a"); head(fig, H, 8, 74.6, "Accuracy by category")
    ax = axmm(fig, H, 78, 13, 40, 52)
    dumbbell(ax, prob, lambda v: f"{v:.2f}", "Mean p(manipulated)", (-.02, 1.04), thr=True); ax.set_yticklabels([])
    letter(fig, H, 64, 75, "b"); head(fig, H, 70, 74.6, "Mean predicted probability")
    ax = axmm(fig, H, 132, 13, 28, 52)
    mets = [("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("mcc", "MCC")]
    for m in ["Full_SSL", "NoSSL"]:
        for key, lab in mets:
            a, b = c56[m]["orig"][key], c56[m]["norm"][key]
            ax.plot([0, 1], [a, b], color=COL[m], lw=.9, alpha=.8, marker="o", ms=2.8)
            if m == "NoSSL": ax.text(1.1, b, lab, va="center", fontsize=5.6, clip_on=False)
    ax.set_xticks([0, 1], ["original", "H.264\nCRF 18"]); ax.set_xlim(-.2, 1.05); ax.set_ylim(.2, 1.03)
    ax.set_ylabel("Test-set metric"); ygrid(ax)
    letter(fig, H, 122, 75, "c"); head(fig, H, 128, 74.6, "Overall metrics")
    fig.legend([Line2D([], [], ls="none", marker="o", mfc="white", mec=MUTED, ms=4), Line2D([], [], ls="none", marker="o", color=MUTED, ms=4),
                Line2D([], [], color=COL["Full_SSL"], lw=1.2), Line2D([], [], color=COL["NoSSL"], lw=1.2),
                Line2D([], [], color=MUTED, lw=.7, ls=(0, (3, 2)))],
               ["original encoding", "re-encoded (H.264 CRF 18, AAC 16 kHz)", "Full SSL", "No SSL", "frozen threshold"],
               frameon=False, ncol=5, loc="center", bbox_to_anchor=(.5, 2.6 / H), handlelength=1.6, columnspacing=1.6)
    save(fig, P, "Figure9_Full_Test_Codec_Normalization")

# =============================================================================
# Figure 10 — protocol-matched comparison
# =============================================================================
def fig10_comparison(P):
    perf = pd.read_csv(P.tables / "Final_Publication_Tables" / "Table_02_Frozen_Validation_Threshold_Performance.csv").set_index("Model")
    rule = codec_rule(pd.read_csv(P.tables / "smdnet_v2_metadata_shortcut_indicators.csv"))
    c56 = load_cell56(P)
    rows = [("Majority class", .5, 0.0, "#9A9A9A"), ("Codec rule (no training)", bal_acc(*rule), mcc(*rule), FAKE)]
    rows += [(NAME[m], float(perf.loc[m, "Balanced_Accuracy"]), float(perf.loc[m, "MCC"]), COL[m]) for m in ["Visual_Only", "Full_SSL", "NoSSL"]]
    rows += [(f"{NAME[m]}, re-encoded test set", c56[m]["norm"]["balanced_accuracy"], c56[m]["norm"]["mcc"], light)
             for m, light in [("Full_SSL", "#B8A9DE"), ("NoSSL", "#9CCBAE")]]
    fig, H = new_fig(62)
    for k, (title, idx, lim) in enumerate([("Balanced accuracy", 1, (.4, 1.05)), ("Matthews correlation coefficient", 2, (0, 1.12))]):
        ax = axmm(fig, H, 48 + k * 68, 10, 52, 44)
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            ax.barh(y, r[idx], color=r[3], height=.62); ax.text(r[idx] + .01, y, f"{r[idx]:.3f}", va="center", fontsize=5.8)
        ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows] if k == 0 else [""] * len(rows))
        ax.tick_params(axis="y", length=0); ax.set_xlim(*lim); ax.set_title(title, fontsize=7, pad=4)
        ax.axhline(1.5, color="#BBBBBB", lw=.6, ls=(0, (3, 2))); xgrid(ax)
        letter(fig, H, 40 + k * 68, 59, "ab"[k])
    save(fig, P, "Figure10_Protocol_Matched_Comparison")

# =============================================================================
# Locate the results package automatically
# =============================================================================
def _is_results(p):
    return (p / "02_Tables").is_dir() and (p / "01_Figures").is_dir()

def _search_dirs(base, max_depth):
    base = Path(base)
    if not base.is_dir():
        return
    stack = [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        for e in entries:
            if e.is_dir() and not e.name.startswith("."):
                yield e
                if depth + 1 < max_depth:
                    stack.append((e, depth + 1))

def find_results(given):
    """Return a folder that contains 02_Tables and 01_Figures, unzipping the package if needed."""
    g = Path(given)
    for cand in [g] + ([c for c in g.iterdir() if c.is_dir()] if g.is_dir() else []):
        if _is_results(cand):
            return cand
    roots = [("/content", 3), ("/content/drive/MyDrive", 4)]
    print("   searching for the results package (folder with 02_Tables and 01_Figures) ...")
    for root, depth in roots:
        for d in _search_dirs(root, depth):
            if d.name == "02_Tables" and _is_results(d.parent):
                print(f"   found: {d.parent}")
                return d.parent
    for root, depth in roots:
        if not Path(root).is_dir():
            continue
        for z in Path(root).glob("**/*Publication_Results*.zip") if depth > 3 else Path(root).glob("*Publication_Results*.zip"):
            dest = Path("/content/SMDNet_V2_All_Publication_Results")
            print(f"   found zip: {z}\n   unzipping to {dest} ...")
            shutil.unpack_archive(str(z), str(dest))
            for cand in [dest] + [c for c in dest.iterdir() if c.is_dir()]:
                if _is_results(cand):
                    return cand
    return None

# =============================================================================
FIGS = {2: fig2_samples, 3: fig3_distribution, 4: fig4_training, 5: fig5_roc, 6: fig6_confusion,
        7: fig7_ablation, 8: fig8_codec, 9: fig9_fulltest, 10: fig10_comparison}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.environ.get("SMDNET_RESULTS", RESULTS_DIR))
    ap.add_argument("--project", default=os.environ.get("SMDNET_PROJECT", PROJECT_DIR))
    ap.add_argument("--notebook", default=os.environ.get("SMDNET_NOTEBOOK", NOTEBOOK))
    ap.add_argument("--out", default=os.environ.get("SMDNET_FIG_OUT", OUTPUT_DIR))
    ap.add_argument("--figures", nargs="*", type=int, default=FIGURES)
    a, _ = ap.parse_known_args()          # ignores Jupyter's own "-f kernel.json" argument
    if a.project and not Path(a.project).is_dir():
        note(f"project folder {a.project} not found; Figs. 2, 5 fall back to published versions.")
        a.project = None
    if 8 in a.figures:
        nb = find_notebook(a.notebook, a.project)
        a.notebook = str(nb) if nb else None
    found = find_results(a.results)
    if found is None:
        raise SystemExit("Results package not found. Upload or unzip SMDNet_V2_All_Publication_Results.zip and set "
                         "RESULTS_DIR to the folder that contains 02_Tables and 01_Figures.")
    a.results = str(found)
    print(f"Results package: {found}")
    P = Paths(a)
    failed = []
    for n in a.figures:
        print(f"Figure {n}: {FIGS[n].__name__}")
        try:
            FIGS[n](P)
        except Exception as e:
            plt.close("all"); failed.append(n)
            print(f"   ERROR in Figure {n}: {type(e).__name__}: {e}")
    print(f"Done. Files in {P.out.resolve()}")
    if failed:
        print(f"Figures that failed: {failed} (see the errors above)")

if __name__ == "__main__":
    main()

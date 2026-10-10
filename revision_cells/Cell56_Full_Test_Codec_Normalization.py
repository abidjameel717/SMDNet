# =============================================================================
# CELL 56 — FULL-TEST-SET CODEC NORMALISATION (ALL 3,174 TEST VIDEOS)
# SMD-Net V2 | Scientific Reports
#
# Replaces the 48-video diagnostic with the complete frozen test set.
#   1. Restores FakeAVCeleb (KaggleHub, same handle as before)
#   2. Re-encodes each test video exactly as Cell 45:
#        H.264 libx264, preset medium, CRF 18, yuv420p + AAC 192k, 16 kHz mono
#   3. Runs Full_SSL and NoSSL with the ORIGINAL protocol (Cell 52):
#        code-cell-49 definitions, 5 windows, CUDA AMP,
#        window starts from the ORIGINAL file's duration
#   4. Compares with the FROZEN original predictions (no re-inference needed;
#      Cell 52 verified they reproduce to <= 2.4e-8)
#   5. Paired statistics, per-category tables, figure
#
# RESUMABLE: progress saved to Drive every batch. After a disconnect,
# just rerun this cell. NO TRAINING.
# Expected time on a T4 runtime: ~1.5-3 h (mostly ffmpeg re-encoding).
# =============================================================================

import ast, json, os, re, shutil, subprocess, sys, time, hashlib, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix)
from google.colab import drive

warnings.filterwarnings("ignore", category=FutureWarning)
T0 = time.time()
drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/SMDNet_V2_ScientificReports")
SPLIT_FILE = ROOT / "02_Splits/fakeavceleb_source_disjoint_split_seed42_media_audited.csv"
PRED_DIR = ROOT / "05_Predictions"
OUT = ROOT / "12_Full_Test_Codec_Normalization"
TABLES, FIGS = OUT / "Tables", OUT / "Figures"
for d in [OUT, TABLES, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "Cell56_normalized_video_predictions.csv"
LOCAL_NORM = Path("/content/cell56_normalized")
LOCAL_NORM.mkdir(exist_ok=True)

assert torch.cuda.is_available(), "Enable a T4 GPU: Runtime > Change runtime type."
DEVICE = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

CATEGORIES = {"RealVideo-RealAudio": "RVRA", "RealVideo-FakeAudio": "RVFA",
              "FakeVideo-RealAudio": "FVRA", "FakeVideo-FakeAudio": "FVFA"}
THRESHOLDS = {"Full_SSL": 0.75732421875, "NoSSL": 0.556396484375}
FROZEN = {"Full_SSL": ("smdnet_v2_final_test_5window_predictions.csv", "probability_manipulated"),
          "NoSSL":    ("smdnet_v2_nossl_test_5window_predictions.csv", "nossl_probability")}
CHECKPOINTS = {"Full_SSL": ROOT / "04_Checkpoints/Full_SSL_Initialized_Fast/smdnet_v2_full_best.pt",
               "NoSSL":    ROOT / "04_Checkpoints/NoSSL_Fast/smdnet_v2_nossl_best.pt"}
BATCH = 48            # videos re-encoded in parallel, then inferred, then deleted
WORKERS = max(2, os.cpu_count() or 2)

def to_relative(p):
    p = str(p).replace("\\", "/")
    for c in CATEGORIES:
        k = p.find(c + "/")
        if k >= 0:
            return p[k:]
    return p.lstrip("/")

def find_col(frame, names):
    low = {str(c).lower(): c for c in frame.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"None of {names} in {list(frame.columns)}")

# -----------------------------------------------------------------------------
# 1. Frozen test split + frozen original predictions
# -----------------------------------------------------------------------------
split = pd.read_csv(SPLIT_FILE)
split["rel"] = split[find_col(split, ["relative_path", "video_path", "path"])].map(to_relative)
split_col = find_col(split, ["split", "subset", "partition"])
test = split[split[split_col].astype(str).str.lower().str.strip() == "test"][["rel"]].copy()
test["label"] = (~test["rel"].str.startswith("RealVideo-RealAudio/")).astype(int)
test["category"] = test["rel"].str.split("/").str[0].map(CATEGORIES)
test = test.reset_index(drop=True)
assert len(test) == 3174, f"Expected 3,174 test videos, found {len(test)}"

for model, (fname, col) in FROZEN.items():
    pred = pd.read_csv(PRED_DIR / fname)
    pred["rel"] = pred[find_col(pred, ["relative_path", "video_path", "path"])].map(to_relative)
    test = test.merge(pred[["rel", col]].rename(columns={col: f"{model}_original"}),
                      on="rel", how="left")
    assert test[f"{model}_original"].notna().all(), f"{model}: frozen predictions missing"
print("Test videos:", len(test), "| frozen original predictions aligned for both models")

# -----------------------------------------------------------------------------
# 2. Original model + decoder definitions (code cell 49, as in Cells 48-52)
# -----------------------------------------------------------------------------
notebooks = sorted(ROOT.rglob("Video_forgery*.ipynb"),
                   key=lambda p: (p.name != "Video_forgery (1).ipynb", -p.stat().st_size))
assert notebooks, "Original notebook not found in the project folder."
NOTEBOOK = notebooks[0]
code_cells = ["".join(c.get("source", [])) for c in
              json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c.get("cell_type") == "code"]
REQUIRED = ["preprocess_rgb_frame", "decode_visual_opencv_exact", "decode_visual_ffmpeg_exact_window",
            "decode_visual_segment", "decode_audio_segment",
            "VisualEncoder", "AudioEncoder", "ProjectionHead", "FusionClassifier", "SMDNetV2"]
defs = {n.name: n for n in ast.parse(code_cells[49]).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
missing = [n for n in REQUIRED if n not in defs]
assert not missing, f"Code cell 49 is missing {missing}; wrong notebook revision?"

WINDOW_SEC, NUM_FRAMES = 0.8, 16
AUDIO_SR, AUDIO_SAMPLES = 16000, 12800
N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS = 640, 400, 160, 128
IMAGE_SIZE, RESIZE_SHORT = 224, 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=AUDIO_SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=0.0, f_max=8000.0, power=2.0, center=True)
ffmpeg_path = shutil.which("ffmpeg")
assert ffmpeg_path, "FFmpeg unavailable."
DECODER_STATS = {"opencv_success": 0, "ffmpeg_fallback": 0, "complete_failure": 0}
WINDOW_FRACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

for name in REQUIRED:
    mod = ast.Module(body=[defs[name]], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, f"original_cell49:{name}", "exec"), globals())

models = {}
for name, ckpt_path in CHECKPOINTS.items():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt.get("test_data_used") is False, f"{name}: provenance check failed"
    m = SMDNetV2(pretrained_visual=False)
    m.load_state_dict(ckpt["model_state_dict"], strict=True)
    models[name] = m.to(DEVICE).eval()
    print(name, "checkpoint loaded (strict) | epoch", ckpt.get("epoch"))

# -----------------------------------------------------------------------------
# 3. Restore FakeAVCeleb originals
# -----------------------------------------------------------------------------
try:
    import kagglehub
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "kagglehub"], check=True)
    import kagglehub

def valid_root(root, k=100):
    sample = test["rel"].sample(min(k, len(test)), random_state=0)
    return root.is_dir() and np.mean([(root / r).is_file() for r in sample]) >= 0.98

DATA_ROOT = None
print("\nRestoring FakeAVCeleb with KaggleHub (cached after the first download) ...")
base = Path(kagglehub.dataset_download("shreyaty08/fakeavceleb"))
for cand in [base / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2", base / "FakeAVCeleb_v1.2", base]:
    if valid_root(cand):
        DATA_ROOT = cand
        break
assert DATA_ROOT is not None, f"Could not find the test videos under {base}"
print("DATA_ROOT:", DATA_ROOT)

# -----------------------------------------------------------------------------
# 4. Re-encoding (identical to Cell 45) and original-protocol inference
# -----------------------------------------------------------------------------
def probe(path, entries):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)

def duration_of(path):
    return float(probe(path, "format=duration")["format"]["duration"])

def normalize(rel):
    src = DATA_ROOT / rel
    dst = LOCAL_NORM / (hashlib.md5(rel.encode()).hexdigest() + ".mp4")
    if not (dst.is_file() and dst.stat().st_size > 1000):
        cmd = [ffmpeg_path, "-y", "-v", "error", "-i", str(src),
               "-map", "0:v:0", "-map", "0:a:0",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-ar", "16000", "-ac", "1",
               "-movflags", "+faststart", "-threads", "1", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return rel, None, None, r.stderr[-300:]
    streams = probe(dst, "stream=codec_type,codec_name,sample_rate")["streams"]
    v = [s for s in streams if s["codec_type"] == "video"]
    a = [s for s in streams if s["codec_type"] == "audio"]
    if not v or v[0]["codec_name"] != "h264" or not a or a[0]["codec_name"] != "aac" \
            or str(a[0].get("sample_rate")) != "16000":
        return rel, None, None, "normalised stream check failed"
    starts = WINDOW_FRACTIONS * max(0.0, duration_of(src) - WINDOW_SEC)  # ORIGINAL duration
    return rel, dst, starts, None

def infer(path, starts):
    vis = [decode_visual_segment(path, float(s), WINDOW_SEC, NUM_FRAMES) for s in starts]
    aud = [decode_audio_segment(path, float(s), WINDOW_SEC) for s in starts]
    out = {}
    with torch.inference_mode():
        vb, ab = torch.stack(vis).to(DEVICE), torch.stack(aud).to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=True):      # Full_SSL: 5-window batch
            out["Full_SSL"] = torch.sigmoid(models["Full_SSL"](vb, ab)["logit"]).float().cpu().numpy().reshape(-1)
        probs = []
        for v, a in zip(vis, aud):                                       # NoSSL: per window
            with torch.amp.autocast("cuda", enabled=True):
                probs.append(float(torch.sigmoid(models["NoSSL"](v.unsqueeze(0).to(DEVICE),
                                                                 a.unsqueeze(0).to(DEVICE))["logit"]).cpu().item()))
        out["NoSSL"] = np.asarray(probs)
    for k, p in out.items():
        assert len(p) == 5 and np.isfinite(p).all(), f"{k}: bad window output"
    return {k: float(np.mean(p)) for k, p in out.items()}

records = pd.read_csv(PROGRESS).to_dict("records") if PROGRESS.is_file() else []
done = {r["rel"] for r in records}
todo = [r for r in test["rel"] if r not in done]
print(f"\nAlready done: {len(done)} | remaining: {len(todo)} | workers: {WORKERS}")

def save():
    tmp = Path("/content/cell56_progress.tmp.csv")
    pd.DataFrame(records).to_csv(tmp, index=False)
    shutil.copy2(tmp, PROGRESS)

t_start, n_new = time.time(), 0
for b in range(0, len(todo), BATCH):
    chunk = todo[b:b + BATCH]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        prepared = list(ex.map(normalize, chunk))
    for rel, dst, starts, err in prepared:
        if err:
            records.append({"rel": rel, "status": "failed", "error": err})
            continue
        try:
            probs = infer(dst, starts)
            records.append({"rel": rel, "status": "ok",
                            "Full_SSL_normalized": probs["Full_SSL"], "NoSSL_normalized": probs["NoSSL"]})
        except Exception as e:
            records.append({"rel": rel, "status": "failed", "error": repr(e)[:300]})
        finally:
            dst.unlink(missing_ok=True)
        n_new += 1
    save()
    rate = (time.time() - t_start) / max(n_new, 1)
    left = len(todo) - (b + len(chunk))
    print(f"  {len(records)}/{len(test)} done | {rate:.2f} s/video | ETA {left * rate / 60:.0f} min")

# -----------------------------------------------------------------------------
# 5. Analysis
# -----------------------------------------------------------------------------
res = pd.DataFrame(records)
failed = res[res["status"] != "ok"]
print("\nFailed videos:", len(failed))
if len(failed):
    failed.to_csv(TABLES / "Cell56_failed_videos.csv", index=False)
data = test.merge(res[res["status"] == "ok"].drop(columns=["status"], errors="ignore"), on="rel", how="inner")
print("Paired videos analysed:", len(data), "of", len(test))

def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y, yhat), "roc_auc": roc_auc_score(y, p),
            "specificity": tn / (tn + fp), "sensitivity": tp / (tp + fn),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}

rng = np.random.default_rng(42)
y = data["label"].values
idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
overall, by_cat, tests = [], [], []
for model, thr0 in THRESHOLDS.items():
    po, pn = data[f"{model}_original"].values, data[f"{model}_normalized"].values
    for policy, thr in [("Frozen_Validation", thr0), ("Default_0.5", 0.5)]:
        mo, mn = metrics(y, po, thr), metrics(y, pn, thr)
        overall += [{"model": model, "policy": policy, "encoding": "Original", **mo},
                    {"model": model, "policy": policy, "encoding": "Normalized_H264_CRF18", **mn}]
        co, cn = (po >= thr).astype(int) == y, (pn >= thr).astype(int) == y
        b_lost, b_gained = int((co & ~cn).sum()), int((~co & cn).sum())
        p_mc = binomtest(min(b_lost, b_gained), b_lost + b_gained, 0.5).pvalue if b_lost + b_gained else 1.0
        boots_acc, boots_bal = [], []
        for _ in range(5000):                                   # stratified paired bootstrap
            s = np.r_[rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))]
            boots_acc.append(cn[s].mean() - co[s].mean())
            yo, yn = (po[s] >= thr).astype(int), (pn[s] >= thr).astype(int)
            boots_bal.append(balanced_accuracy_score(y[s], yn) - balanced_accuracy_score(y[s], yo))
        tests.append({"model": model, "policy": policy,
                      "accuracy_change_pp": 100 * (mn["accuracy"] - mo["accuracy"]),
                      "accuracy_change_ci95_pp": tuple(np.round(100 * np.percentile(boots_acc, [2.5, 97.5]), 2)),
                      "balanced_accuracy_change_pp": 100 * (mn["balanced_accuracy"] - mo["balanced_accuracy"]),
                      "balanced_accuracy_change_ci95_pp": tuple(np.round(100 * np.percentile(boots_bal, [2.5, 97.5]), 2)),
                      "correct_to_wrong": b_lost, "wrong_to_correct": b_gained, "exact_mcnemar_p": p_mc})
        for cat in ["RVRA", "RVFA", "FVRA", "FVFA"]:
            m = data["category"].values == cat
            by_cat.append({"model": model, "policy": policy, "category": cat, "n": int(m.sum()),
                           "correct_original": int(co[m].sum()), "correct_normalized": int(cn[m].sum()),
                           "accuracy_original": co[m].mean(), "accuracy_normalized": cn[m].mean(),
                           "mean_prob_original": po[m].mean(), "mean_prob_normalized": pn[m].mean()})

overall, by_cat, tests = pd.DataFrame(overall), pd.DataFrame(by_cat), pd.DataFrame(tests)
# Holm across the 4 tests (2 models x 2 policies)
order = np.argsort(tests["exact_mcnemar_p"].values)
holm, running = np.empty(len(tests)), 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(tests) - rank) * tests["exact_mcnemar_p"].values[i]))
    holm[i] = running
tests["holm_p"] = holm

overall.to_csv(TABLES / "Cell56_overall_metrics.csv", index=False)
by_cat.to_csv(TABLES / "Cell56_per_category.csv", index=False)
tests.to_csv(TABLES / "Cell56_paired_tests.csv", index=False)
data.to_csv(TABLES / "Cell56_paired_video_predictions.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, model in zip(axes, THRESHOLDS):
    sub = by_cat[(by_cat.model == model) & (by_cat.policy == "Frozen_Validation")]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["accuracy_original"], 0.4, label="Original encoding", color="#4C72B0")
    ax.bar(x + 0.2, sub["accuracy_normalized"], 0.4, label="H.264 CRF 18", color="#DD8452")
    ax.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(sub.category, sub.n)])
    ax.set_title(model)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Accuracy (frozen threshold)")
axes[0].legend(loc="lower left")
fig.suptitle("Full test set: accuracy before and after codec normalisation")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(FIGS / f"Figure_Cell56_Full_Test_Codec_Normalization.{ext}", dpi=300)
plt.show()

pd.set_option("display.width", 250)
print("\n" + "=" * 100 + "\nCELL 56 RESULTS (paste this output back)\n" + "=" * 100)
print(overall[["model", "policy", "encoding", "accuracy", "balanced_accuracy", "macro_f1", "mcc",
               "roc_auc", "specificity", "sensitivity", "fp", "fn"]].round(4).to_string(index=False))
print("\nPaired tests:\n", tests.round(6).to_string(index=False))
print("\nPer category (frozen threshold):\n",
      by_cat[by_cat.policy == "Frozen_Validation"].round(4).to_string(index=False))
print(f"\nSaved to {OUT}\nTotal time {(time.time() - T0) / 60:.1f} min | Training: NO")

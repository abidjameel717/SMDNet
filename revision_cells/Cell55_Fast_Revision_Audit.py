# =============================================================================
# CELL 55 — FAST REVISION AUDIT (NO TRAINING, NO GPU)
# SMD-Net V2 | FakeAVCeleb v1.2 | Scientific Reports
#
#  A. Metadata-only shortcut baseline (ffprobe features -> simple classifiers)
#  B. Identity-clean test subset (re-scores your EXISTING predictions)
#  C. Imbalance-aware metrics + majority-class baseline for every model
#
# Needs: the restored FakeAVCeleb video files (same as your recovery cells).
# Runtime: ~10-20 min for ffprobe on 21,476 videos (cached after first run).
# =============================================================================

import json, re, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from google.colab import drive
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix)

START = time.time()
drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/SMDNet_V2_ScientificReports")
SPLIT_FILE = ROOT / "02_Splits/fakeavceleb_source_disjoint_split_seed42_media_audited.csv"
PRED_DIR = ROOT / "05_Predictions"
OUT = ROOT / "11_Fast_Revision"
OUT.mkdir(parents=True, exist_ok=True)

PRED_FILES = {
    "Full_SSL":    ("smdnet_v2_final_test_5window_predictions.csv", "probability_manipulated", 0.75732421875),
    "NoSSL":       ("smdnet_v2_nossl_test_5window_predictions.csv", "nossl_probability", 0.556396484375),
    "Visual_Only": ("smdnet_v2_visual_only_test_5window_predictions.csv", "visual_only_probability", 0.9638671875),
}
CATEGORIES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
              "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_relative(p):
    """Normalise any absolute/relative path to start at the category folder."""
    p = str(p).replace("\\", "/")
    for c in CATEGORIES:
        k = p.find(c + "/")
        if k >= 0:
            return p[k:]
    return p.lstrip("/")

def find_col(frame, candidates):
    lookup = {str(c).lower(): c for c in frame.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None

def metrics(y, prob, thr):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "n_authentic": int((y == 0).sum()), "n_manipulated": int((y == 1).sum()),
        "threshold": thr,
        "accuracy": accuracy_score(y, pred),
        "majority_baseline_accuracy": max(y.mean(), 1 - y.mean()),
        "balanced_accuracy": balanced_accuracy_score(y, pred) if both else np.nan,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if both else np.nan,
        "roc_auc": roc_auc_score(y, prob) if both else np.nan,
        # PR-AUC for the RARE (authentic) class: the honest imbalance metric
        "pr_auc_authentic": average_precision_score(1 - y, 1 - prob) if both else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

# -----------------------------------------------------------------------------
# Load split
# -----------------------------------------------------------------------------
df = pd.read_csv(SPLIT_FILE)
path_col = find_col(df, ["relative_path", "video_path", "path"])
split_col = find_col(df, ["split", "subset", "partition"])
df["rel"] = df[path_col].map(to_relative)
df["split"] = (df[split_col].astype(str).str.lower().str.strip()
               .replace({"training": "train", "validation": "val", "valid": "val", "testing": "test"}))
df["label"] = (~df["rel"].str.startswith("RealVideo-RealAudio/")).astype(int)
df["category"] = df["rel"].str.split("/").str[0]
print("Split counts:\n", df.groupby(["split", "label"]).size().unstack())

# -----------------------------------------------------------------------------
# Locate dataset root
# -----------------------------------------------------------------------------
# A valid root must actually contain the split's videos (checked on a sample),
# not just a folder named RealVideo-RealAudio (e.g. the 4-video samples folder).
def root_is_valid(root, k=200):
    sample = df["rel"].sample(min(k, len(df)), random_state=0)
    return sum((root / r).is_file() for r in sample) / len(sample) >= 0.95

DATA_ROOT = None
# Local disk only: never crawl Google Drive (slow, and holds only sample videos).
search_bases = [Path("/root/.cache/kagglehub"), Path("/content/fakeavceleb"),
                Path("/content/data"), Path("/content/datasets")]
for base in search_bases:
    if not base.exists():
        continue
    for cand in base.rglob("RealVideo-RealAudio"):
        if cand.is_dir() and "Representative" not in str(cand) and root_is_valid(cand.parent):
            DATA_ROOT = cand.parent
            break
    if DATA_ROOT:
        break
RUN_METADATA = DATA_ROOT is not None
print("DATA_ROOT:", DATA_ROOT if RUN_METADATA else
      "NOT FOUND -> Part A (metadata baseline) skipped; Parts B and C still run.")

# =============================================================================
# A. METADATA-ONLY SHORTCUT BASELINE
# =============================================================================
CACHE = OUT / "Cell55_ffprobe_metadata.csv"
rows = []
data = df.copy()
tr, va, te = (data["split"] == s for s in ["train", "val", "test"])
y = data["label"].values

if RUN_METADATA:
    def probe(rel):
        path = DATA_ROOT / rel
        cmd = ["ffprobe", "-v", "error", "-show_entries",
               "stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels,bit_rate"
               ":format=duration,size,bit_rate", "-of", "json", str(path)]
        row = {"rel": rel}
        try:
            info = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
        except Exception as e:
            row["probe_error"] = str(e)
            return row
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and "v_codec" not in row:
                num, _, den = str(s.get("avg_frame_rate", "0/1")).partition("/")
                row.update(v_codec=s.get("codec_name"), width=s.get("width"), height=s.get("height"),
                           fps=float(num) / float(den or 1) if float(den or 1) else np.nan,
                           v_bitrate=s.get("bit_rate"))
            elif s.get("codec_type") == "audio" and "a_codec" not in row:
                row.update(a_codec=s.get("codec_name"), a_sample_rate=s.get("sample_rate"),
                           a_channels=s.get("channels"), a_bitrate=s.get("bit_rate"))
        f = info.get("format", {})
        row.update(duration=f.get("duration"), file_size=f.get("size"), total_bitrate=f.get("bit_rate"))
        return row

    meta = None
    if CACHE.is_file():
        meta = pd.read_csv(CACHE)
        if "v_codec" not in meta or meta["v_codec"].isna().mean() > 0.01:
            print("Discarding invalid cache from a failed run:", CACHE)
            CACHE.unlink()
            meta = None
        else:
            print("Loaded cached ffprobe metadata:", len(meta))
    if meta is None:
        print("Running ffprobe on", len(df), "videos ...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            meta = pd.DataFrame(list(ex.map(probe, df["rel"].tolist())))
        fail_rate = meta["v_codec"].isna().mean() if "v_codec" in meta else 1.0
        assert fail_rate <= 0.01, f"ffprobe failed on {fail_rate:.1%} of videos; check DATA_ROOT"
        meta.to_csv(CACHE, index=False)

    data = df.merge(meta, on="rel", how="left")
    assert len(data) == len(df)
    print("Probe failures:", int(data["v_codec"].isna().sum()))

    NUM = ["width", "height", "fps", "v_bitrate", "a_sample_rate", "a_channels",
           "a_bitrate", "duration", "file_size", "total_bitrate"]
    CAT = ["v_codec", "a_codec"]
    for c in NUM:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    X_all = pd.get_dummies(data[NUM + CAT].astype({c: "string" for c in CAT}),
                           columns=CAT, dummy_na=True).fillna(-1).astype(float)

    tr, va, te = (data["split"] == s for s in ["train", "val", "test"])
    y = data["label"].values

    baselines = {
        "Metadata_LogReg": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=5000, class_weight="balanced")),
        "Metadata_Tree_depth3": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
    }
    for name, model in baselines.items():
        model.fit(X_all[tr], y[tr])
        p_val = model.predict_proba(X_all[va])[:, 1]
        p_te = model.predict_proba(X_all[te])[:, 1]
        # validation-only threshold (max balanced accuracy), mirroring your protocol
        grid = np.unique(np.r_[p_val, 0.5])
        val_thr = max(grid, key=lambda t: balanced_accuracy_score(y[va], (p_val >= t).astype(int)))
        rows.append({"model": name, "policy": "Default_0.5", **metrics(y[te], p_te, 0.5)})
        rows.append({"model": name, "policy": "Frozen_Validation", **metrics(y[te], p_te, val_thr)})

    # Single hand-written rule: "not H.264 => fake"
    codec_rule = (data.loc[te, "v_codec"].astype(str) != "h264").astype(float).values
    rows.append({"model": "Rule_video_codec_not_h264", "policy": "Rule", **metrics(y[te], codec_rule, 0.5)})

    tree_rules = export_text(baselines["Metadata_Tree_depth3"], feature_names=list(X_all.columns))
    (OUT / "Cell55A_metadata_tree_rules.txt").write_text(tree_rules)

    # Codec-by-label cross tables for the paper
    for split in ["train", "val", "test"]:
        ct = pd.crosstab(data.loc[data["split"] == split, "v_codec"],
                         data.loc[data["split"] == split, "label"]).rename(columns={0: "authentic", 1: "manipulated"})
        ct.to_csv(OUT / f"Cell55A_codec_by_label_{split}.csv")

# =============================================================================
# B + C. RE-SCORE EXISTING MODEL PREDICTIONS (full test + identity-clean subset)
# =============================================================================
def ids_in(rel):
    return set(re.findall(r"id\d{5}", rel))

train_ids = set().union(*data.loc[tr, "rel"].map(ids_in))
test = data.loc[te, ["rel", "label", "category"]].copy()
test["all_ids_unseen_in_train"] = test["rel"].map(lambda r: ids_in(r).isdisjoint(train_ids))
# Authentic test videos are kept (source-disjoint by protocol); manipulated videos
# are kept only if NO identity in their path (source or target) appears in training.
test["identity_clean"] = (test["label"] == 0) | test["all_ids_unseen_in_train"]
print("\nIdentity-clean subset:")
print(test.groupby(["category", "identity_clean"]).size().unstack(fill_value=0))
print("Authentic test videos whose identity appears anywhere in training:",
      int(((test["label"] == 0) & ~test["all_ids_unseen_in_train"]).sum()))

for model, (fname, prob_col, frozen_thr) in PRED_FILES.items():
    pred = pd.read_csv(PRED_DIR / fname)
    pcol = find_col(pred, ["relative_path", "video_path", "path"])
    pred["rel"] = pred[pcol].map(to_relative)
    m = test.merge(pred[["rel", prob_col]], on="rel", how="inner")
    assert len(m) == len(test), f"{model}: {len(m)} of {len(test)} predictions matched"
    for subset_name, mask in [("Full_test", np.ones(len(m), bool)),
                              ("Identity_clean", m["identity_clean"].values)]:
        for policy, thr in [("Default_0.5", 0.5), ("Frozen_Validation", frozen_thr)]:
            rows.append({"model": model, "subset": subset_name, "policy": policy,
                         **metrics(m.loc[mask, "label"], m.loc[mask, prob_col], thr)})

results = pd.DataFrame(rows)
if "subset" not in results:
    results["subset"] = "Full_test"
results["subset"] = results["subset"].fillna("Full_test")
cols = ["model", "subset", "policy", "n", "n_authentic", "n_manipulated", "accuracy",
        "majority_baseline_accuracy", "balanced_accuracy", "macro_f1", "mcc", "roc_auc",
        "pr_auc_authentic", "tn", "fp", "fn", "tp", "threshold"]
results = results[cols]
results.to_csv(OUT / "Cell55_Fast_Revision_Results.csv", index=False)
test.to_csv(OUT / "Cell55B_identity_clean_test_membership.csv", index=False)

pd.set_option("display.width", 250)
print("\n" + "=" * 100)
print("CELL 55 RESULTS")
print("=" * 100)
print(results.round(4).to_string(index=False))
if RUN_METADATA:
    print("\nDepth-3 metadata tree rules:\n", tree_rules)
print("\nSaved to:", OUT)
print(f"Done in {(time.time() - START) / 60:.1f} min | Training: NO | Download: NO")

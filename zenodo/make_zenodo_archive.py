#!/usr/bin/env python3
"""
make_zenodo_archive.py — package the SMD-Net V2 research artefacts for Zenodo.

Run in Google Colab after mounting Drive. It collects the project folder
(splits, checkpoints, predictions, metadata audits, result tables, figures),
EXCLUDES all FakeAVCeleb video/audio files (the dataset may not be redistributed),
writes a SHA-256 manifest, and creates zip parts below Zenodo's 50 GB record limit.

    1. Set PROJECT_DIR / OUTPUT_DIR below (or pass --project / --out).
    2. Run once with DRY_RUN = True to review what will be included.
    3. Set DRY_RUN = False and run again to build the archive.
    4. Upload everything in OUTPUT_DIR to a new Zenodo record (see UPLOAD_GUIDE.md).
"""
import argparse, csv, hashlib, json, os, time, zipfile
from pathlib import Path

# ----------------------------------------------------------------------------- settings
PROJECT_DIR = "/content/drive/MyDrive/SMDNet_V2_ScientificReports"
OUTPUT_DIR = "/content/zenodo_upload"      # local disk is faster than Drive; copy to Drive afterwards if needed
DRY_RUN = True
MAX_PART_GB = 45                            # Zenodo allows 50 GB per record

EXCLUDED_DIRS = {"08_Representative_Dataset_Samples", ".ipynb_checkpoints", "__pycache__", ".Trash", "kagglehub"}
EXCLUDED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wav", ".mp3", ".aac", ".m4a", ".flac", ".f32"}
TITLE = ("Beyond Benchmark Accuracy: Investigating Encoding Bias and Robustness in Audiovisual "
         "Deepfake Detection — data, checkpoints and audit outputs")

# ----------------------------------------------------------------------------- helpers
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def collect(root):
    keep, skipped = [], {"excluded folder": 0, "media file (dataset)": 0, "duplicate zip": 0}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            skipped["excluded folder"] += 1; continue
        if p.suffix.lower() in EXCLUDED_EXT:
            skipped["media file (dataset)"] += 1; continue
        if p.suffix.lower() == ".zip" and (p.parent / p.stem).is_dir():
            skipped["duplicate zip"] += 1; continue          # the unzipped folder is already included
        keep.append(p)
    return keep, skipped

README = """# {title}

Research artefacts accompanying the manuscript
"Beyond Benchmark Accuracy: Investigating Encoding Bias and Robustness in Audiovisual Deepfake Detection"
(Mazhar, Elmannai, Jameel, Alsehaim, Hayat, Serat and Khan; submitted to Scientific Reports).

Code: [GitHub URL] (archived at [software DOI]).

## Contents
{inventory}

Every file is listed in MANIFEST.csv with its size, SHA-256 checksum and the zip part that holds it.
Verify downloads with:  python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" FILE

## Not included: FakeAVCeleb videos
No FakeAVCeleb video or audio file is redistributed here. Obtain FakeAVCeleb v1.2 from its authors
(Khalid et al., NeurIPS Datasets and Benchmarks 2021) under their licence; the split files in 02_Splits
identify every video by its relative path and SHA-256 hash, so the exact cohort can be rebuilt.

## Licence
Data, predictions, tables and figures: Creative Commons Attribution 4.0 (CC BY 4.0).
Model checkpoints: CC BY 4.0, derived from training on FakeAVCeleb; users must also respect the dataset licence.
Code: see the GitHub repository (MIT).
"""

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("SMDNET_PROJECT", PROJECT_DIR))
    ap.add_argument("--out", default=os.environ.get("SMDNET_ZENODO_OUT", OUTPUT_DIR))
    ap.add_argument("--build", action="store_true", help="build the archive (default follows DRY_RUN)")
    a, _ = ap.parse_known_args()
    root, out = Path(a.project), Path(a.out)
    if not root.is_dir():
        raise SystemExit(f"Project folder not found: {root}  (mount Drive and check PROJECT_DIR)")
    dry = DRY_RUN and not a.build

    files, skipped = collect(root)
    sizes = {p: p.stat().st_size for p in files}
    total = sum(sizes.values())
    by_top = {}
    for p in files:
        top = p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else "(top level)"
        n, s = by_top.get(top, (0, 0)); by_top[top] = (n + 1, s + sizes[p])
    print(f"Project: {root}\nFiles to include: {len(files):,}  ({human(total)})")
    for k, (n, s) in sorted(by_top.items()):
        print(f"   {k:<45s} {n:>7,} files  {human(s):>10s}")
    print("Skipped:", ", ".join(f"{k}: {v:,}" for k, v in skipped.items()))
    big = [p for p in files if sizes[p] > MAX_PART_GB * 1024 ** 3]
    if big:
        raise SystemExit(f"Files larger than {MAX_PART_GB} GB cannot be packed: {big}")
    if dry:
        print("\nDRY RUN — nothing written. Review the list above, then set DRY_RUN = False (or pass --build).")
        return

    out.mkdir(parents=True, exist_ok=True)
    limit = MAX_PART_GB * 1024 ** 3
    parts, cur, cur_size = [], [], 0
    for p in files:
        if cur and cur_size + sizes[p] > limit:
            parts.append(cur); cur, cur_size = [], 0
        cur.append(p); cur_size += sizes[p]
    if cur:
        parts.append(cur)

    manifest, t0 = [], time.time()
    for i, part in enumerate(parts, start=1):
        name = f"SMDNet_V2_artefacts_part{i:02d}.zip" if len(parts) > 1 else "SMDNet_V2_artefacts.zip"
        zpath = out / name
        print(f"\nWriting {name} ({len(part):,} files) ...")
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for k, p in enumerate(part, start=1):
                rel = p.relative_to(root).as_posix()
                z.write(p, arcname=rel)
                manifest.append({"path": rel, "bytes": sizes[p], "sha256": sha256(p), "zip_part": name})
                if k % 500 == 0:
                    print(f"   {k:,}/{len(part):,} files  ({(time.time() - t0) / 60:.1f} min)")
        print(f"   done: {human(zpath.stat().st_size)}")

    with open(out / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "zip_part"]); w.writeheader(); w.writerows(manifest)
    inventory = "\n".join(f"- `{k}/`: {n:,} files, {human(s)}" for k, (n, s) in sorted(by_top.items()))
    (out / "README_ZENODO.md").write_text(README.format(title=TITLE, inventory=inventory), encoding="utf-8")
    zips = sorted(out.glob("*.zip"))
    (out / "zip_checksums.sha256").write_text("".join(f"{sha256(z)}  {z.name}\n" for z in zips))
    json.dump({"title": TITLE, "files": len(manifest), "bytes": total, "parts": [z.name for z in zips]},
              open(out / "archive_summary.json", "w"), indent=2)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Upload everything in {out} to Zenodo:")
    for p in sorted(out.iterdir()):
        print(f"   {p.name:<45s} {human(p.stat().st_size):>10s}")

if __name__ == "__main__":
    main()

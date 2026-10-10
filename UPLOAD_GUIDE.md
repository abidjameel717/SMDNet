# Upload guide: GitHub and Zenodo

You will create **two Zenodo records**, each with its own DOI:

| Record | What | How |
|---|---|---|
| **Software** | this repository (code, notebooks, figure scripts) | automatically, from a GitHub release |
| **Data** | checkpoints, splits, predictions, tables (several GB) | manual upload of the zip made by `zenodo/make_zenodo_archive.py` |

Do the data record first, so you can reserve its DOI and put it in the manuscript before submission.

---

## Part 1 — Data record on Zenodo

1. **Build the archive in Colab.** Mount Drive, upload `zenodo/make_zenodo_archive.py` and run:
   ```python
   !python make_zenodo_archive.py --project "/content/drive/MyDrive/SMDNet_V2_ScientificReports" --out /content/zenodo_upload
   ```
   This is a dry run: check the list of folders and the "Skipped" line (dataset videos must be skipped). Then build:
   ```python
   !python make_zenodo_archive.py --project "/content/drive/MyDrive/SMDNet_V2_ScientificReports" --out /content/zenodo_upload --build
   ```
   It writes the zip part(s), `MANIFEST.csv`, `README_ZENODO.md`, `zip_checksums.sha256` and `archive_summary.json`. Copy `/content/zenodo_upload` to Drive if the upload will happen later (`!cp -r /content/zenodo_upload /content/drive/MyDrive/`).

2. **Create the record.** Log in at https://zenodo.org → **New upload**.
   - Click **Reserve DOI** (under "Digital Object Identifier") *before* uploading. Copy this DOI into the manuscript's Data availability statement and the cover letter.
   - Upload every file from `zenodo_upload/`. Zenodo accepts up to 50 GB per record; the script keeps each zip below 45 GB.

3. **Fill in the metadata.**
   - Resource type: **Dataset**
   - Title: *Beyond Benchmark Accuracy: Investigating Encoding Bias and Robustness in Audiovisual Deepfake Detection — data, checkpoints and audit outputs*
   - Creators: the seven authors, in manuscript order (add ORCID iDs and affiliations if available)
   - Description: paste the first paragraphs of `README_ZENODO.md`
   - Licence: **Creative Commons Attribution 4.0 International**
   - Keywords: deepfake detection; audiovisual forensics; FakeAVCeleb; shortcut learning; dataset bias; video compression
   - Funding: Princess Nourah bint Abdulrahman University, project PNURSP2026R972
   - Related works: *is supplement to* the article (add its DOI after publication); *is documented by* the GitHub repository URL

4. **Publish** (or keep it as a draft and share the reserved DOI; a published record cannot be deleted, only versioned).

---

## Part 2 — Code on GitHub

1. Create a new **public** repository at https://github.com/new, for example `SMDNet-V2`. Do not add a README or licence there (they are in this package).
2. Upload the files, either:
   - **In the browser:** *Add file → Upload files*, drag in the whole contents of the `SMDNet-V2` folder (including the hidden files `.gitignore` and `.zenodo.json`; on macOS press Cmd + Shift + . to show them), then *Commit changes*; or
   - **With git:**
     ```bash
     cd SMDNet-V2
     git init && git add . && git commit -m "SMD-Net V2 code for the Scientific Reports submission"
     git branch -M main
     git remote add origin https://github.com/USERNAME/SMDNet-V2.git
     git push -u origin main
     ```
3. Replace the placeholders `[USERNAME]`, `[DATA DOI]` and `[SOFTWARE DOI]` in `README.md` and `CITATION.cff` (edit directly on GitHub).
4. GitHub now shows a **"Cite this repository"** button, generated from `CITATION.cff`.

---

## Part 3 — Software DOI from a GitHub release

1. On Zenodo: profile menu → **GitHub** → **Sync now**, then switch the toggle **on** for `SMDNet-V2`.
2. On GitHub: **Releases → Draft a new release** → tag `v1.0.0`, title *SMD-Net V2 v1.0.0 (Scientific Reports submission)* → **Publish release**.
3. Within a few minutes Zenodo archives the release and mints a DOI, using the metadata in `.zenodo.json`. Copy the DOI badge from the Zenodo record into `README.md`.

Note: Zenodo can only archive releases made *after* the toggle is switched on.

---

## Part 4 — Update the manuscript

Replace the placeholders in the manuscript, cover letter and README:

- Data availability → the **data** record DOI
- Code availability → the GitHub URL and the **software** record DOI
- Cover letter → both

## Checklist before publishing

- [ ] Dry run shows no `.mp4` / `.wav` files and no `08_Representative_Dataset_Samples` folder
- [ ] `zip_checksums.sha256` verifies after upload (`sha256sum -c zip_checksums.sha256`)
- [ ] Both DOIs are in the manuscript and cover letter
- [ ] `requirements.txt` updated with `pip freeze` from your Colab runtime if you want exact pins
- [ ] All co-authors agree with the licences (MIT for code, CC BY 4.0 for data)

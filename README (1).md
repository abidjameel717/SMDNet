# SMD-Net V2 — Beyond Benchmark Accuracy

Code for the manuscript:

> **Beyond Benchmark Accuracy: Investigating Encoding Bias and Robustness in Audiovisual Deepfake Detection**
> Ahmad A. Mazhar, Hela Elmannai, Abid Jameel, Aishah Alsehaim, Faisal Hayat, Ali Mohammad Serat, Shaukat Hayat and Iftikhar Ali Khan.
> *Scientific Reports* (submitted, 2026).

SMD-Net V2 is a self-supervised audiovisual deepfake detector (CNN–Transformer visual encoder, CNN–BiLSTM audio encoder, contrastive pretraining and supervised fusion; 28.4 M parameters). The paper evaluates it on FakeAVCeleb v1.2 under a frozen, audited protocol and shows that near-perfect benchmark accuracy partly reflects video encoding: a training-free codec rule reaches 0.973 balanced accuracy, and re-encoding the test set to H.264 lowers the detector's accuracy by 22–24 percentage points while ROC-AUC stays near 0.97.

[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23288367.svg)](https://doi.org/10.5281/zenodo.23288367) [![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23289450.svg)](https://doi.org/10.5281/zenodo.23289450)

**Data, checkpoints and predictions:** [doi.org/10.5281/zenodo.23288367](https://doi.org/10.5281/zenodo.23288367) · **This code:** [doi.org/10.5281/zenodo.23289450](https://doi.org/10.5281/zenodo.23289450)

## Repository structure

```
SMDNet/
├── notebooks/
│   ├── SMDNet_V2_Reproduction.ipynb       clean, linear notebook — start here
│   └── Video_forgery_development.ipynb    complete development notebook with all outputs
├── revision_cells/
│   ├── Cell55_Fast_Revision_Audit.py      identity-clean subset, imbalance-aware metrics
│   ├── Cell56_Full_Test_Codec_Normalization.py   all 3,174 test videos re-encoded to H.264
│   └── Cell57_Compute_Cost.py             parameters, FLOPs and inference time
├── figures/
│   ├── make_figure1.py                    Figure 1 (methodology)
│   ├── make_figures.py                    Figures 2–10
│   └── output/                            the figures as published in the manuscript
├── zenodo/
│   └── make_zenodo_archive.py             packages the data artefacts for Zenodo
├── requirements.txt · LICENSE · CITATION.cff · .zenodo.json
└── UPLOAD_GUIDE.md                        step-by-step GitHub and Zenodo instructions
```

## Quick start (Google Colab, T4 GPU)

1. Download the data record from Zenodo ([10.5281/zenodo.23288367](https://doi.org/10.5281/zenodo.23288367)) and place its contents in `MyDrive/SMDNet_V2_ScientificReports` (or point the notebooks at another folder).
2. Download FakeAVCeleb v1.2 from its authors; the notebooks fetch it through KaggleHub (`shreyaty08/fakeavceleb`). Its use is subject to the dataset licence.
3. Open `notebooks/SMDNet_V2_Reproduction.ipynb`:
   - **Part A** re-derives every reported number from the frozen artefacts (CPU is enough; minutes).
   - **Part B** runs the codec-control experiments and compute measurement (T4; about 4–6 h).
   - **Part C** retrains everything from raw data (T4; about 10–14 h).

The development notebook is kept for transparency and is required by the codec experiments, which load the exact model definitions from its code cell 49. Its first 91 cells are the original study; cells 91–102 were added during revision (Cells 55–57, figures and export).

## Regenerating the figures

```bash
pip install -r requirements.txt
python figures/make_figure1.py --results path/to/SMDNet_V2_All_Publication_Results --out figures_out
python figures/make_figures.py --results path/to/SMDNet_V2_All_Publication_Results \
       --project path/to/SMDNet_V2_ScientificReports --out figures_out
```

Both scripts also run when pasted into a Colab cell: edit the settings at the top. `make_figures.py` finds the results package and the development notebook automatically; if an optional input is missing it says so and uses the values verified for the manuscript. Each figure is written as PDF, PNG and TIFF (600 dpi, 180 mm wide).

## Frozen protocol

| Item | Value |
|---|---|
| Cohort | 21,476 videos: 21,560 files − 51 label-conflict duplicates − 17 same-label duplicates − 16 without metadata; all decodable |
| Split | Source-identity-disjoint, seed 42: 15,065 / 3,237 / 3,174 videos (350 / 75 / 75 identities); 0 shared identities or SHA-256 hashes |
| Thresholds (validation only) | Full_SSL 0.75732421875 · NoSSL 0.556396484375 · Visual_Only 0.9638671875 |
| Test inference | 5 windows (0, ¼, ½, ¾, 1 of duration), mean probability, CUDA mixed precision |
| Optimiser | AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 1e-4, cosine annealing; seed 42, deterministic cuDNN |
| Codec normalisation | libx264 CRF 18, preset medium, yuv420p; AAC 192 kb/s, 16 kHz, mono |

## Where each result comes from

Cell numbers are those printed in the notebooks; development-notebook indices are in parentheses.

| Manuscript item | Source |
|---|---|
| Table 2, training settings | Cells 18, 21, 23D (22, 28, 37) |
| Table 3 and Fig. 3, cohort and split | Cells 4–11 (3–10) |
| Table 5, test performance | Cell 27 (61) |
| Table 6, class-wise precision, recall, F1 | Cell 27 confusion counts; Cell 55 |
| Table 7, accuracy by category | Cell 27 (61) |
| Table 8, paired comparisons | Cell 37 (73) |
| Table 9, encoding attributes | Cell 28 (62) |
| Table 10 and Fig. 9, full-test codec normalisation | Cell 56 |
| Table 11 and Fig. 8c–d, 48-video re-encoding | Cells 45, 48–53 (81, 84–89) |
| Fig. 8e–f, Visual_Only CRF sweep | Cell 35 (71) |
| Table 12, compute cost | Cell 57 |
| Identity leakage | Cell 29 (63), Cell 55 |
| Reproducibility checks | Cells 30–32, 51–52 (64–66, 87–88) |
| Figures 1–10 | `figures/make_figure1.py`, `figures/make_figures.py` |

## Known limitations

- Part C's cell order was reconstructed from the development history and checked by static dependency analysis, not by one uninterrupted run. After a Colab disconnect, run the runtime bootstrap cell (Appendix) and re-run the interrupted cell; training resumes from checkpoints.
- Retraining will not reproduce the checkpoints bit for bit (GPU nondeterminism). Inference with the frozen checkpoints and CUDA AMP reproduces the frozen probabilities to within 2.4 × 10⁻⁸.
- All models were trained once (seed 42). The leading-silence artefact in FakeAVCeleb's fake audio (Smeu et al., arXiv:2412.00175) was not trimmed.

## Licence and citation

Code: MIT (see `LICENSE`). Data, predictions and checkpoints on Zenodo: CC BY 4.0. FakeAVCeleb is not redistributed and remains under its authors' licence. Please cite the article above (see `CITATION.cff`).

**Use of AI tools:** Claude (Anthropic) helped draft the manuscript and write the revision cells and figure scripts; the authors ran all experiments and verified every result.

**Funding:** Princess Nourah bint Abdulrahman University Researchers Supporting Project number (PNURSP2026R972), Princess Nourah bint Abdulrahman University, Riyadh, Saudi Arabia.

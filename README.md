# 🌴 Date Fruit Analyzer

Detects clusters of dates on the palm and classifies each fruit by **variety**
and **ripeness stage**, using three custom-trained YOLOv8 models chained
together: detection → crop → variety classification → maturity classification.

Originally built as a graduation project (PFE) on automating date harvesting
in the Drâa-Tafilalet region of Morocco; rebuilt here as a Streamlit app.

> This was a two-person graduation project, co-authored with **Bouhani
> ELMustafa** under the supervision of Pr. Mohamed OUHDA (École Supérieure de
> Technologie de Khénifra, 2023–2024). The dataset collection, model
> training, and original Flask interface were joint work; this Streamlit
> rebuild of the interface was done independently afterward.

## What it does

1. **Detection** — locates date clusters in an orchard photo (YOLOv8, trained
   on ~9,000 in-orchard images across two Moroccan orchards).
2. **Variety classification** — labels each detected fruit as one of
   `Boufagous`, `Bouisthami`, `Boumajhoul`, or `Kholt`.
3. **Maturity classification** — labels each fruit's ripeness stage as
   `Kimri` (unripe), `Khalal`, `Rutab`, or `Tamar` (fully ripe).
4. Renders an annotated image with per-fruit results, and keeps a lightweight
   local history of past analyses.

## Results (from the thesis)

- Detection: **92.7% mAP@0.5** on the held-out test set.
- Variety classification accuracy per class (normalized confusion matrix):
  Boufagous 97%, Bouisthami 100%, Boumajhoul 94%, Kholt 96%.
- Full methodology, dataset breakdown, and training curves are in
  [`Rapport_PFE.pdf`](./Rapport_PFE.pdf).

## Running it locally

```bash
git clone <this-repo>
cd date-fruit-analyzer
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Model weights (`models/Detection.pt`, `models/Variety.pt`, `models/Maturity.pt`)
are included directly in the repo, so no separate download step is needed.

### A note on model weights

`Variety.pt` is ~70MB — under GitHub's 100MB hard limit, but past the 50MB
mark where GitHub shows a "large file" warning on push. That's just a
notice, not a block, and since these weights are final (not actively
retrained), committing them directly keeps things simple: clone and run,
no extra setup step. If the models grow past 100MB in the future, or you
start retraining and versioning them regularly, revisit this with
[Git LFS](https://git-lfs.com/) or a GitHub Release instead.

## Dataset

The training data ("Date Fruit Detection Dataset for Automatic Harvesting")
was collected specifically for this project and published on Zenodo — see
the thesis for the exact citation. The raw training images aren't
redistributed in this repo — only the final trained model weights, which
are committed directly under `models/`.

## Tech

`ultralytics` (YOLOv8) · `streamlit` · `Pillow` · `pandas` · `sqlite3` for
lightweight local history (no server-side database needed).

## Project structure

```
app.py                 # Streamlit app — single entry point
models/                # Detection.pt, Variety.pt, Maturity.pt — committed directly
requirements.txt
history.db             # created at runtime, not versioned
history_outputs/        # annotated images from past analyses, not versioned
```

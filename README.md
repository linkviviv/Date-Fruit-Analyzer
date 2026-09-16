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

## Why this project

Drâa-Tafilalet produces roughly 85% of Morocco's dates — about 94,000 tonnes
a year across 48,000+ hectares  and the harvest is still done almost
entirely by hand, with pickers judging ripeness fruit-by-fruit. This project
explores automating that judgment call: detecting date clusters in orchard
photos and classifying each fruit's variety and ripeness stage, as a step
toward vision-guided harvesting robots.

## What it does

1. **Detection** — locates date clusters in an orchard photo (YOLOv8, trained
   on ~9,000 in-orchard images across two Moroccan orchards).
2. **Variety classification** — labels each detected fruit as one of
   `Boufagous`, `Bouisthami`, `Boumajhoul`, or `Kholt`.
3. **Maturity classification** — labels each fruit's ripeness stage as
   `Kimri` (unripe), `Khalal`, `Rutab`, or `Tamar` (fully ripe).
4. Renders an annotated image with per-fruit results, and keeps a lightweight
   local history of past analyses.

> **Naming note:** the classification labels use `Boumajhoul` (as trained
> into the model), while the dataset folders and thesis tables refer to the
> same variety as `Majhoul` 

## Dataset

Collected specifically for this project between June and September 2022, in
two orchards in Morocco, using a Canon camera and a Samsung Ultra smartphone.
Annotated in Roboflow and exported in YOLO format. Published on Zenodo — see
the thesis for the exact citation; raw images aren't redistributed in this
repo, only the final trained weights.

| | Images | Clusters | Palms |
|---|---|---|---|
| **Boufagous** | 4,325 | 50 | 10 |
| **Majhoul** | 2,645 | 58 | 14 |
| **Bouisthami** | 360 | 14 | 2 |
| **Kholt** | 1,762 | 6 | 2 |
| **Total** | **9,092** | **128** | **28** |

Each variety is also labeled across four maturity stages  Immature (Kimri),
Khalal, Rutab, and Tamar  captured under varying light, angle, and
tree-bag-covering conditions to reflect real orchard variability.

## Training

- Detection and both classification heads trained with **YOLOv8**
  (Ultralytics) via Google Colab, using a Tesla P4 GPU.
- Dataset organized in Ultralytics' standard `train/val/test` split format,
  with a `data.yaml` defining the single `Date_fruit` detection class.
- Classification heads trained on per-class image folders (one per variety,
  one per maturity stage).

## Results 

- Detection: **92.7% mAP@0.5** on the held-out test set; per the normalized
  confusion matrix, ~90% of true date-fruit clusters were correctly detected
  (the remaining ~10% missed as background).
- Variety classification accuracy per class (normalized confusion matrix):
  Boufagous 97%, Bouisthami 100%, Boumajhoul 94%, Kholt 96%.
- Classification training curves show top-1 accuracy climbing from ~75% to
  ~95% over training, with top-5 accuracy constant at 100%.
- Full methodology, dataset breakdown, and training curves are in
  [`Rapport_PFE.pdf`](./Rapport_PFE.pdf).

## Limitations (from the thesis)

- The dataset covers 4 date varieties, not the full range grown in the
  region — the model's adaptability to other varieties is untested.
- No seasonal variation in date appearance is accounted for; all images were
  collected within a single June–September harvest window.

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

## Tech

`ultralytics` (YOLOv8) · `streamlit` · `Pillow` · `pandas` · `sqlite3` for
lightweight local history (no server-side database needed). Original
training was done in Python 3.7 via Google Colab and Google Drive; data
annotation via Roboflow.

## Project structure

```
app.py                 # Streamlit app — single entry point
models/                # Detection.pt, Variety.pt, Maturity.pt — committed directly
requirements.txt
history.db             # created at runtime, not versioned
history_outputs/        # annotated images from past analyses, not versioned
```

# survey-ml-risk

Interpretable machine-learning models using population survey data for chronic disease risk prediction: a scalable framework for preventive health.

**Authors:** Reagan M. Mogire, Daniel Shriner, Charles N. Rotimi, Adebowale A. Adeyemo  
**Affiliation:** Center for Research on Genomics and Global Health, NHGRI/NIH, Bethesda, MD, USA

---

## Overview

This repository provides a reproducible pipeline for training and evaluating **survey-only** chronic disease risk models using BRFSS (2011–2015), plus a Streamlit demonstration app.

### Core idea
Most ML risk models rely on EHRs or biomarkers. Here, models are trained using only population survey variables, enabling scalable, low-cost risk stratification for prevention and public health planning (with appropriate cautions).

---

## Repository structure

```
app/                    # Streamlit app and deployment artifacts
  app.py
  artifacts/            # downloaded model artifacts (from GitHub Releases)
data/
  raw/                  # downloaded BRFSS year extracts (.csv.zip) (from GitHub Releases)
  derived/              # derived pooled dataset (optional; from GitHub Releases or built locally)
docs/                   # optional documentation
figures/                # manuscript figures (PDF)
manuscript/             # manuscript files (optional)
scripts/
  fetch_release_assets.py
src/
  data_preparation.py   # constructs pooled modeling dataset from BRFSS annual extracts
  model_ai.py           # training + evaluation + artifact export
tables/                 # manuscript tables (CSV)
```

**Large files (data + model artifacts)** are distributed via **GitHub Releases** to keep the git repository lightweight and fast to clone.

---

## Releases

This project uses two releases:

- **data-v1**: BRFSS annual extracts (2011–2015) + optional pooled cleaned dataset (`BRFSS_2011_2015_clean_model.csv.gz`)
- **model-v1**: serialized trained models and metadata (`*.joblib`) used by the Streamlit app

Download them using:

```bash
python scripts/fetch_release_assets.py --data-tag data-v1 --model-tag model-v1
```

---

## Installation

### 1) Clone
```bash
git clone https://github.com/reaganmogire/survey-ml-risk.git
cd survey-ml-risk
```

### 2) Environment
Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
python -m pip install -U pip
```

### 3) Dependencies

Create a `requirements.txt` at repo root (recommended for Streamlit Cloud). Minimum:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- shap
- matplotlib
- requests

Then install:

```bash
pip install -r requirements.txt
```

---

## Reproducibility: data + artifacts

### Download everything (data + models)
```bash
python scripts/fetch_release_assets.py --data-tag data-v1 --model-tag model-v1
```

### Download only model artifacts (fast; sufficient for Streamlit app)
```bash
python scripts/fetch_release_assets.py --only-model --model-tag model-v1
```

### Download only data assets
```bash
python scripts/fetch_release_assets.py --only-data --data-tag data-v1
```

---

## Build the pooled modeling dataset (optional)

If you want to regenerate `BRFSS_2011_2015_clean_model.csv` from the raw annual extracts:

```bash
python scripts/fetch_release_assets.py --only-data --data-tag data-v1
python src/data_preparation.py
```

The preprocessing script reads `.csv.zip` files directly; manual extraction is not required.

If you download the precomputed pooled dataset (`.gz`) from the release and want to decompress:

```bash
gunzip data/derived/BRFSS_2011_2015_clean_model.csv.gz
```

---

## Train models + produce outputs

```bash
python src/model_ai.py
```

Expected outputs:
- `app/artifacts/disease_models.joblib`
- `app/artifacts/optimal_thresholds.joblib`
- `app/artifacts/predictor_cols.joblib`
- manuscript-ready figures under `figures/`
- manuscript-ready tables under `tables/`

---

## Run the Streamlit app locally

```bash
streamlit run app/app.py
```

The app expects model artifacts under `app/artifacts/`. If missing:

```bash
python scripts/fetch_release_assets.py --only-model --model-tag model-v1
```

---

## Deploy on Streamlit Cloud

1. Streamlit Cloud → **New app**
2. Select:
   - Repo: `reaganmogire/survey-ml-risk`
   - Branch: `main`
   - Main file path: `app/app.py`
3. Ensure `requirements.txt` exists at the repo root.
4. Recommended: configure the app to auto-download `model-v1` artifacts at startup (keeps the deploy light).

---

## Reproducibility and code availability (manuscript-ready text)

All data preprocessing, model development, evaluation, and deployment workflows were implemented in Python (v3.11) using open-source libraries (e.g., pandas, NumPy, scikit-learn, SHAP). The complete analytical pipeline, including preprocessing scripts, model training and evaluation code, and versioned model artefacts, is publicly available at:

https://github.com/reaganmogire/survey-ml-risk

To support reproducibility while keeping the repository lightweight, large input datasets and model artefacts are distributed via GitHub Releases (tags: `data-v1`, `model-v1`) and can be retrieved using `scripts/fetch_release_assets.py`. An interactive demonstration interface implementing the trained models is available via a Streamlit web deployment (URL to be added).

---

## Disclaimer

**Disclaimer: This tool is for research only. It does not provide medical advice or diagnosis. Consult a qualified clinician for personalised guidance.**

---

## License

See `LICENSE`.


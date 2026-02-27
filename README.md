# Machine-Learning Models for Chronic Disease Risk Prediction 

### Interpretable Machine-Learning Models for Chronic Disease Risk Prediction Using Population Survey Data

**Authors:**\
Reagan M. Mogire, Daniel Shriner, Charles N. Rotimi, Adebowale A.
Adeyemo\
Center for Research on Genomics and Global Health\
National Human Genome Research Institute (NHGRI), NIH

------------------------------------------------------------------------

## Overview

This repository provides a fully reproducible implementation of an
interpretable machine-learning framework for predicting chronic disease
risk using large-scale population survey data.

Unlike many risk models that rely on electronic health records,
laboratory biomarkers, or clinical imaging, this framework operates
exclusively on self-reported survey variables, enabling scalable
deployment in resource-limited settings.

The models were trained using the U.S. Behavioral Risk Factor
Surveillance System (BRFSS) 2011--2015 dataset (N ≈ 2.38 million
adults).

------------------------------------------------------------------------

## Objectives

-   Develop interpretable gradient-boosted decision tree models\
-   Provide well-calibrated risk estimates across multiple chronic
    diseases\
-   Enable reproducibility via versioned data and model releases\
-   Provide a lightweight Streamlit interface for demonstration and
    research

------------------------------------------------------------------------

## Predicted Conditions

-   Myocardial infarction (heart attack)\
-   Coronary heart disease\
-   Stroke\
-   Chronic kidney disease\
-   Diabetes\
-   Depression

------------------------------------------------------------------------

## Model Performance (Test Set)

Cardiometabolic outcomes achieved AUROC values in the range 0.79--0.86.\
Calibration was assessed using Brier score and calibration curves.

Full evaluation tables are available in the `/tables` directory.

------------------------------------------------------------------------

## Repository Structure

app/ Streamlit application\
src/ Data preparation and model training scripts\
scripts/ Release asset download utilities\
data/ (empty; populated via GitHub Releases)\
figures/ Publication-ready figures\
tables/ Publication-ready result tables

Large datasets and trained models are stored as versioned GitHub
Releases:

-   `data-v1` → BRFSS 2011--2015 zipped extracts\
-   `model-v1` → Trained model artifacts

------------------------------------------------------------------------

## Reproducibility

### 1. Install dependencies

pip install -r requirements.txt

### 2. Download assets

python scripts/fetch_release_assets.py

### 3. Recreate cleaned dataset

python src/data_preparation.py

### 4. Retrain models

python src/model_ai.py

### 5. Launch interactive app

streamlit run app/app.py

------------------------------------------------------------------------

## Deployment

The application is deployable on Streamlit Community Cloud.

Main entry point:

app/app.py

On first run, model artifacts are automatically downloaded from GitHub
Releases.

------------------------------------------------------------------------

## Interpretability

Local explanations are generated using SHAP for tree-based classifiers
within sklearn Pipelines.

Feature contributions reflect post-preprocessing transformed features.

------------------------------------------------------------------------

## Limitations

-   Based on self-reported survey data\
-   U.S. population (BRFSS 2011--2015)\
-   Not externally validated in non-U.S. cohorts\
-   Not designed for clinical deployment

------------------------------------------------------------------------

## Disclaimer

Disclaimer: This tool is for research only. It does not provide medical
advice or diagnosis. Consult a qualified clinician for personalised
guidance.

------------------------------------------------------------------------

## Citation

If using this repository, please cite:

Mogire RM et al.\
Interpretable machine-learning models using population survey data for
chronic disease risk prediction: a scalable framework for preventive
health.

(Manuscript under review)

------------------------------------------------------------------------

## License

(Add LICENSE file -- MIT or Apache 2.0 recommended)

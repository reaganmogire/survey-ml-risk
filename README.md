# Survey-ML Risk  
### Interpretable Machine-Learning Models for Chronic Disease Risk Prediction Using Population Survey Data

**Authors:**  
Reagan M. Mogire, Daniel Shriner, Charles N. Rotimi, Adebowale A. Adeyemo  
Center for Research on Genomics and Global Health  
National Human Genome Research Institute (NHGRI), NIH  

---

## Overview

This repository provides a fully reproducible implementation of an interpretable machine-learning framework for predicting chronic disease risk using large-scale population survey data.

Unlike many risk models that rely on electronic health records, laboratory biomarkers, or clinical imaging, this framework operates exclusively on self-reported survey variables, enabling scalable deployment in resource-limited settings.

The models were trained using the U.S. Behavioral Risk Factor Surveillance System (BRFSS) 2011–2015 dataset (N ≈ 2.38 million adults).

---

## Objectives

- Develop interpretable gradient-boosted decision tree models
- Provide well-calibrated risk estimates across multiple chronic diseases
- Enable reproducibility via versioned data and model releases
- Provide a lightweight Streamlit interface for demonstration and research

---

## Predicted Conditions

- Myocardial infarction (heart attack)
- Coronary heart disease
- Stroke
- Chronic kidney disease
- Diabetes
- Depression

---

## Model Performance (Test Set)

| Condition | AUROC Range |
|-----------|-------------|
| Cardiometabolic outcomes | 0.79–0.86 |
| Depression | Lower but clinically informative |

Calibration was assessed using Brier score and calibration curves.

Full evaluation tables are available in the `/tables` directory.

---

## Repository Structure

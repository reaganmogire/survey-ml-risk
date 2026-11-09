# survey-ml-risk

Code, analysis scripts and outputs for:

> Mogire RM, Shriner D, Rotimi CN, Adeyemo AA. *Survey-Based Interpretable Machine Learning for Detecting Prevalent Chronic Disease in 2.38 Million US Adults: A Model Development Study.* [Citation and DOI to be added on publication.]

Six gradient-boosted classifiers detect prevalent myocardial infarction, coronary heart disease, stroke, chronic kidney disease, diabetes and lifetime history of diagnosed depression from self-reported survey responses (BRFSS 2011-2015, 2.38 million US adults). The repository covers data preparation, model training and evaluation, SHAP explanations, subgroup analyses and the interactive application.

Live demonstration: https://healthymeai.streamlit.app/

## Repository layout

```
app/       Streamlit application (app.py) and configuration
src/       data_preparation.py, model_ai.py and the Fig 5 flow-diagram scripts
scripts/   Analysis scripts that generate the manuscript tables and figures
tables/    CSV outputs: manuscript tables and supplementary datasets
figures/   Manuscript and supplementary figures (PDF and PNG)
data/      raw/ and derived/ (not tracked in git; see Data and models)
```

`app/artifacts/` (trained models) and `data/` are not tracked in git. They are distributed as GitHub Release assets.

## Requirements

Python 3.11 or newer. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Data and models

- **Source data:** the US Behavioral Risk Factor Surveillance System (BRFSS) annual files for 2011-2015. The primary source is the Centers for Disease Control and Prevention (https://www.cdc.gov/brfss); the files used here were obtained through the Kaggle repository (https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system).
- **Release assets:** release `data-v2` contains the raw files (`2011.csv.zip` to `2015.csv.zip`) and the cleaned dataset (`BRFSS_2011_2015_clean_model.csv.gz`). Release `model-v2` contains the trained models (`disease_models.joblib`, `optimal_thresholds.joblib`, `predictor_cols.joblib`). Retrieve them with:

```bash
python scripts/fetch_release_assets.py
```

  See the header of that script for options. Raw files belong in `data/raw/`, the cleaned dataset in `data/derived/`, and the model artifacts in `app/artifacts/`.
- **Path overrides:** the scripts use repository-relative defaults, which can be overridden with environment variables documented at the top of each script (for example `ARTIFACT_DIR`, `TABLES_DIR`, `FIGURES_DIR`, `BRFSS_RAW_DIR`, `BRFSS_DERIVED_DIR`).

## Data preparation notes

`src/data_preparation.py` builds the modelling dataset from the raw BRFSS files:

- Outcomes are binary indicators derived from `CVDINFR4`, `CVDCRHD4`, `CVDSTRK3`, `CHCKIDNY`, `ADDEPEV2` and `DIABETE3`.
- `DIABETE3`: 1 = diabetes; 2 (diabetes only during pregnancy), 3 (no) and 4 (pre-diabetes or borderline) = no diabetes; 7 and 9 = missing.
- Codes 7, 9, 77 and 99 are treated as missing only in columns where they mean "don't know" or "refused". They are real values in some columns (for example age bands, state, and the 0-30 day counts) and are kept there.
- Fruit, vegetable and alcohol frequencies (`FRUIT1`, `VEGETAB1`, `ALCDAY5`) are decoded from BRFSS's compound day/week/month coding to a per-day rate. For the three unhealthy-days variables, "None" (88) is coded as zero days.
- Predictors not administered in some survey years (for example race and ethnicity in 2011-2012) are imputed within the modelling pipeline, as described in the manuscript Methods.

## Model

`src/model_ai.py` trains one `HistGradientBoostingClassifier` per outcome (learning rate 0.1, 300 iterations, no depth limit, `random_state = 42`) inside a scikit-learn pipeline with one-hot encoding of nominal predictors and median imputation and standardisation of numeric predictors. Data are split with stratification into training (64%), validation (16%) and test (20%) sets. The decision threshold is chosen on the validation set by maximising F1.

## Reproducing the analyses

Run from the repository root, in this order:

1. `python scripts/fetch_release_assets.py` (raw data and, optionally, trained models)
2. `python src/data_preparation.py`
3. `python src/model_ai.py`
4. `python scripts/baseline_model_comparison.py`, then `python scripts/retune_random_forest_baseline.py`
5. `python scripts/generate_global_shap.py`, then `python scripts/refresh_flagged_cluster_summary.py`
6. `python scripts/restricted_years_sensitivity.py` (requires step 5)
7. `python scripts/subgroup_fairness_analysis.py`, then `python scripts/subgroup_calibration.py`
8. `python scripts/compute_nns_and_ppv_prevalence.py` and `python scripts/decision_curve_analysis.py`
9. `python scripts/verify_train_test_equivalence.py`, `python scripts/generate_participant_flow.py`, `python scripts/generate_s3_missingness_table.py` and `python scripts/revision_analyses.py --phase 1`
10. `python src/generate_flowchart_data.py`, then `python src/generate_flow_diagram_pdf.py` (Fig 5; the output is copied to `figures/`)
11. `python scripts/make_manuscript_outputs.py` (last: Table 1 and the main-text and supplementary figures)

`revision_analyses.py --phase 1` also writes a calibration-bin diagnostic table that is not cited in the manuscript.

## Mapping of outputs to the manuscript

File names under `tables/` do not always match the manuscript's dataset numbers (for example, `tables/S1_participant_flow_by_year.csv` is S13 Dataset). Use this table.

| Manuscript item | Output file | Generated by |
|---|---|---|
| Table 1 | tables/table3_model_performance_updated.csv | scripts/make_manuscript_outputs.py |
| Table 2 | Literature comparison (compiled manually) | n/a |
| Fig 1 | figures/fig_roc_curves.* | scripts/make_manuscript_outputs.py |
| Fig 2 | figures/fig_model_comparison_all_metrics.* | scripts/make_manuscript_outputs.py (reads output of scripts/baseline_model_comparison.py and retune_random_forest_baseline.py) |
| Fig 3 | figures/fig_global_shap_violin_combined.* | scripts/plot_shap_violin.py (reads output of scripts/generate_global_shap.py) |
| Fig 4 | figures/example_predicted_probabilities.*, example_shap_explanations.* | scripts/make_manuscript_outputs.py |
| Fig 5 | figures/flow_diagram.* | src/generate_flowchart_data.py, src/generate_flow_diagram_pdf.py |
| S1 Table | tables/table1_combined_with_prevalence.csv | scripts/make_manuscript_outputs.py |
| S2 Table | tables/restricted_years_sensitivity/restricted_years_performance_comparison.csv | scripts/restricted_years_sensitivity.py |
| S3 Table | tables/model_comparison_summary_formatted.csv | scripts/make_manuscript_outputs.py |
| S4 Table | tables/ppv_npv_vs_prevalence.csv | scripts/compute_nns_and_ppv_prevalence.py |
| S5 Table | tables/decision_curve_summary.csv | scripts/decision_curve_analysis.py |
| S6 Table | tables/table_example_inputs.csv | scripts/make_manuscript_outputs.py |
| S1 Fig | figures/fig_ppv_vs_prevalence.* | scripts/compute_nns_and_ppv_prevalence.py |
| S2 Fig | figures/fig_decision_curves.* | scripts/decision_curve_analysis.py |
| S3 Fig | figures/fig_global_shap_bar_all_outcomes.* | scripts/make_manuscript_outputs.py |
| S4 Fig | figures/fig_calibration_curves.* | scripts/make_manuscript_outputs.py |
| S5 Fig | Screenshot of the web application | app/app.py |
| S1 Dataset | tables/phase1_3_full_vs_analytic_cohort.csv | scripts/revision_analyses.py (--phase 1) |
| S2 Dataset | tables/tripod_item21_partition_counts.csv | scripts/verify_train_test_equivalence.py |
| S3 Dataset | tables/tripod_item20c_dev_vs_eval_comparison.csv | scripts/verify_train_test_equivalence.py |
| S4 Dataset | tables/restricted_years_sensitivity/restricted_years_shap_rank_stability.csv | scripts/restricted_years_sensitivity.py |
| S5 Dataset | tables/global_shap_top15_all_outcomes.csv | scripts/generate_global_shap.py |
| S6 Dataset | tables/global_shap_full_ranking_all_outcomes.csv | scripts/generate_global_shap.py |
| S7 Dataset | tables/global_shap_flagged_cluster_summary.csv | scripts/generate_global_shap.py, then scripts/refresh_flagged_cluster_summary.py |
| S8 Dataset | tables/fairness/subgroup_fairness_full.csv | scripts/subgroup_fairness_analysis.py |
| S9 Dataset | tables/fairness/subgroup_auroc_gaps_summary.csv | scripts/subgroup_fairness_analysis.py |
| S10, S12 Datasets | TRIPOD+AI checklists | Provided with the article (not in this repository) |
| S11 Dataset | tables/S3_missing_data_predictors.csv | scripts/generate_s3_missingness_table.py |
| S13 Dataset | tables/S1_participant_flow_by_year.csv | scripts/generate_participant_flow.py |
| S14 Dataset | tables/S2_participant_flow_by_outcome.csv | scripts/generate_participant_flow.py |
| S15 Dataset | tables/fairness/subgroup_calibration.csv | scripts/subgroup_calibration.py |

The SHAP summary used for Fig 3 reads per-observation SHAP values written to `tables/global_shap/` by `generate_global_shap.py`. These files are large and are regenerated locally, not stored in git.

## Reproducibility

Random seeds are fixed (`random_state = 42`; bootstrap resampling uses seeded generators), so rerunning the pipeline on the same inputs reproduces the reported results. Training on the full cohort takes substantial time and memory; the trained models are provided as release assets for users who only want to reproduce the tables and figures.

## Versions

- `v1.0-original-submission` (git tag): the code as first submitted. Its data and model assets are in releases `data-v1` and `model-v1`.
- `v2.0-revision` (git tag): the code for the revised manuscript. Data assets are in release `data-v2` and trained models in release `model-v2`.

## Interactive application

`app/app.py` serves the trained models through a Streamlit interface (https://healthymeai.streamlit.app/). Users enter self-reported information and receive predicted probabilities for the six outcomes with SHAP-based explanations. To run it locally after fetching the model artifacts:

```bash
streamlit run app/app.py
```

## Citation and contact

Please cite the article above. Corresponding authors: Charles N. Rotimi (Rotimic@mail.nih.gov) and Adebowale A. Adeyemo (adeyemoa@mail.nih.gov), Center for Research on Genomics and Global Health, National Human Genome Research Institute, National Institutes of Health. Software questions can be raised as GitHub issues.

## License

See the `LICENSE` file in this repository.

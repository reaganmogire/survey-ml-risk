#!/usr/bin/env python
# coding: utf-8
"""
scripts/feature_labels.py

Shared feature-labeling logic for converting post-ColumnTransformer
SHAP/model feature names into human-readable labels, used by both
scripts/make_manuscript_outputs.py and scripts/generate_global_shap.py.

This module was extracted during revision after the same labeling
logic was found duplicated (and had drifted -- one copy had a bug)
across both scripts. Import from here in both places going forward
rather than maintaining separate copies.

Handles two cases:
  1. One-hot encoded nominal categorical column (e.g.
     'cat__TOLDHI2_1.0') -> "Told high cholesterol: Yes"
  2. Numeric (ordinal or continuous) column (e.g. 'num__GENHLTH')
     -> "General health"

If you add a new predictor to the modelling pipeline in
src/model_ai.py, update FEATURE_LABELS (and, if the new predictor is
nominal/categorical, NOMINAL_CATEGORICAL_VARS and
CATEGORY_VALUE_LABELS) here -- this is now the single source of truth
for feature labeling across the analysis pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


# ============================================================
# Variable name -> human-readable label
# ============================================================
FEATURE_LABELS: Dict[str, str] = {
    "GENHLTH":  "General health",
    "PHYSHLTH": "Physical health days",
    "MENTHLTH": "Mental health days",
    "POORHLTH": "Days poor health limited activities",
    "BPHIGH4":  "High blood pressure",
    "BPMEDS":   "Blood pressure medication",
    "TOLDHI2":  "Told high cholesterol",
    "CHOLCHK":  "Cholesterol checked recently",
    "SMOKE100": "Ever smoked 100 cigarettes",
    "SMOKDAY2": "Current smoking frequency",
    "ALCDAY5":  "Alcohol drinking days (per day rate)",
    "DRNKANY5": "Any alcohol use",
    "EXERANY2": "Any exercise in past month",
    "FRUIT1":   "Fruit intake (per day rate)",
    "VEGETAB1": "Vegetable intake (per day rate)",
    "ASTHMA3":  "Asthma",
    "HAVARTH3": "Arthritis",
    "DIFFWALK": "Difficulty walking/climbing",
    "DECIDE":   "Cognitive difficulty (memory/concentration)",
    "PERSDOC2": "Has personal doctor",
    "HLTHPLN1": "Has health insurance",
    "MEDCOST":  "Could not see doctor (cost)",
    "CHECKUP1": "Time since last checkup",
    "_BMI5":    "BMI",
    "WEIGHT2":  "Weight (lb)",
    "HEIGHT3":  "Height (feet and inches code)",
    "_AGEG5YR": "Age group",
    "_EDUCAG":  "Education category",
    "_INCOMG":  "Income category",
    "_MRACE1":  "Race",
    "_HISPANC": "Hispanic ethnicity",
    "_STATE":   "State",
    "SEX":      "Sex",
}

# Nominal categorical variables that go through one-hot encoding in
# the modelling pipeline. MUST match NOMINAL_CATEGORICAL_COLS in
# src/model_ai.py exactly, or labels will silently fall back to raw
# codes for any variable listed there but missing here.
NOMINAL_CATEGORICAL_VARS = {
    "_STATE", "SEX", "_MRACE1", "_HISPANC", "SMOKE100", "SMOKDAY2",
    "DRNKANY5", "EXERANY2", "HLTHPLN1", "PERSDOC2", "MEDCOST",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "ASTHMA3", "HAVARTH3",
    "DIFFWALK", "DECIDE",
}

# ============================================================
# Category-value labels, for readable one-hot feature labels
# (e.g. "Told high cholesterol: Yes" instead of "TOLDHI2 1.0")
# ============================================================
_BINARY_YES_NO = {1: "Yes", 2: "No"}

CATEGORY_VALUE_LABELS: Dict[str, Dict[int, str]] = {
    "SEX":      {1: "Male", 2: "Female"},
    "BPHIGH4":  {1: "Yes", 2: "No", 3: "Borderline/pre-hypertensive", 4: "Yes, during pregnancy only"},
    "BPMEDS":   _BINARY_YES_NO,
    "TOLDHI2":  _BINARY_YES_NO,
    "ASTHMA3":  _BINARY_YES_NO,
    "HAVARTH3": _BINARY_YES_NO,
    "DIFFWALK": _BINARY_YES_NO,
    "DECIDE":   _BINARY_YES_NO,
    "SMOKE100": _BINARY_YES_NO,
    "DRNKANY5": _BINARY_YES_NO,
    "EXERANY2": _BINARY_YES_NO,
    "HLTHPLN1": _BINARY_YES_NO,
    "MEDCOST":  _BINARY_YES_NO,
    "SMOKDAY2": {1: "Every day", 2: "Some days", 3: "Not at all"},
    "PERSDOC2": {1: "Yes, one", 2: "Yes, more than one", 3: "No"},
    "_MRACE1":  {1: "White", 2: "Black", 3: "Native American/Alaska Native", 4: "Asian",
                 5: "Native Hawaiian/Pacific Islander", 6: "Other race", 7: "Multiracial"},
    "_HISPANC": {1: "Hispanic", 2: "Non-Hispanic"},
    # FIPS codes for states appearing in example user profiles
    # (S1 Table); not an exhaustive 50-state mapping. Add codes here
    # if additional example users or states are introduced.
    "_STATE":   {6: "California", 13: "Georgia", 24: "Maryland"},
}


# ============================================================
# NOTE ON BRFSS CODEBOOK VERIFICATION STATUS:
# CATEGORY_VALUE_LABELS above was filled in from standard BRFSS
# codebook convention at the time this module was written, but --
# unlike the FRUIT1/VEGETAB1/ALCDAY5 compound-encoding fix earlier in
# this revision, which was independently verified against official
# CDC calculated-variable documentation -- these category-value
# mappings (esp. _MRACE1's 8 categories and BPHIGH4's 4 categories)
# have NOT been separately verified against the primary CDC codebook.
# Recommend a spot-check against the actual 2011-2015 codebook before
# treating these labels as authoritative for publication.
# ============================================================


def split_categorical_feature(base_name: str) -> Optional[Tuple[str, float]]:
    """
    For a one-hot encoded feature like 'SEX_1.0' or '_MRACE1_2.0',
    splits into (variable_name, category_value). Returns None if
    base_name does not match '<VAR>_<code>' for any known nominal
    categorical variable. Checks longest variable names first since
    variable names may themselves contain underscores (e.g.
    '_MRACE1'), so a plain string split is not sufficient.
    """
    for var in sorted(NOMINAL_CATEGORICAL_VARS, key=len, reverse=True):
        prefix = var + "_"
        if base_name == var:
            return None  # bare variable name, not a one-hot column
        if base_name.startswith(prefix):
            suffix = base_name[len(prefix):]
            try:
                code = float(suffix)
                return (var, code)
            except ValueError:
                continue
    return None


def pretty_feature_from_explainer(name: str) -> str:
    """
    Converts a post-ColumnTransformer feature name into a human-
    readable label. See module docstring for the two handled cases.
    Falls back to a cleaned-up raw name only if neither a categorical
    split nor a direct FEATURE_LABELS match succeeds.
    """
    s = str(name)
    if s.lower().startswith(("num__", "cat__")):
        s = s[5:]

    split = split_categorical_feature(s)
    if split is not None:
        var, code = split
        var_label = FEATURE_LABELS.get(var, var.lstrip("_").replace("_", " "))
        code_int = int(code) if float(code).is_integer() else code
        value_labels = CATEGORY_VALUE_LABELS.get(var, {})
        value_label = value_labels.get(code_int, f"code {code_int}")
        return f"{var_label}: {value_label}"

    bare = s.lstrip("_")
    if s in FEATURE_LABELS:
        return FEATURE_LABELS[s]
    if bare in FEATURE_LABELS:
        return FEATURE_LABELS[bare]
    if f"_{bare}" in FEATURE_LABELS:
        return FEATURE_LABELS[f"_{bare}"]

    return s.replace("_", " ").strip()

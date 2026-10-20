# Multimodal Framework for MCI-to-Alzheimer's Disease Conversion Prediction

Code for the paper **"An Explainable Leakage-Controlled Multimodal Framework for MCI-to-Alzheimer's Disease Conversion Prediction Using Classification and Survival Analysis"** (ICBME). The study uses tabular data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) to predict conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) with Any-Future and fixed-horizon (24/36/60 months) classification and survival analysis (Cox PH, XGBoost Survival), with SHAP interpretation and nested/sensitivity validation.

## Team

| Name | Role | Affiliation |
|---|---|---|
| Hasti Ebrahimzadeh Navabi | Author | Department of Biomedical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran |
| Danial Rahimi | Author | Department of Biomedical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran |
| Pouya Taghipour Langrodi | Author | Amirkabir University of Technology (Tehran Polytechnic); University of Calgary, Canada |
| **Sadegh Madadi** | **Supervisor** (corresponding author) | Pediatric Neurorehabilitation Research Center, University of Social Welfare and Rehabilitation Sciences, Tehran, Iran |

## Repository contents

- `code/` - analysis scripts (cohort construction, predictor extraction, ranking, classification, survival, sensitivity, SHAP).
- `run_pipeline.py` - runs a named stage in a writable `runs/<name>/` workspace (`python run_pipeline.py --list`).
- `requirements.txt` - Python package versions.

**Only code is published here.** ADNI data, derived tables, CSV results and manuscript files are not included. ADNI data are available to approved applicants at https://adni.loni.usc.edu. Place your own ADNI tables under `data/raw/` (git-ignored).

## Status

All analysis scripts are included. A timing audit of the predictor matrix and corrected analyses (strict-date availability, fold-specific feature ranking, corrected IPCW Brier) are in progress, so the reported numbers are development estimates.

## Setup

```bash
pip install -r requirements.txt   # tested with Python 3.13
python run_pipeline.py --list
```

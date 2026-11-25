# QSPR machine learning models for coal based jet fuel properties

Code for the manuscript *Predicting Physical Properties of Coal Based Jet Fuel
via a Quantitative Structure Property Relationship (QSPR) Machine Learning
Framework*.

Four properties are modelled from mole fraction weighted Mordred descriptors of
mixture components: density (293.15 K), viscosity (293.15 K), flash point and
freezing point. The workflow screens roughly 1800 candidate descriptors with
sequential forward selection (SFS), removes redundant descriptors by Pearson
correlation, and benchmarks six regression algorithms: multiple linear
regression (MLR), support vector regression (SVR), Gaussian process regression
(GPR), artificial neural network (ANN), random forest (RF) and gradient boosted
decision trees (GBDT).

## Workflow

```
training workbook (.xlsx)
        |
        |  step1_model_screening.py
        |  column cleaning -> train/test split -> SFS -> Pearson filter
        |  -> standardisation -> grid search -> test set scores
        v
results/seed_42/
    Final_Selected_Descriptors.csv      <- input for steps 2, 3, 4
    model_performance_summary.xlsx
    predictions_detail.xlsx
    Table_S1_<Property>.csv / .xlsx
        |
        +--> step2_loo_cv.py            -> results/LOO_CV_summary.xlsx
        +--> step3_validation.py        -> results/validation_summary.xlsx
        +--> step4_shap_analysis.py     -> results/shap/SHAP_<Property>_<Model>.xlsx
```

The train/test split is created **before** feature selection, and every scaler
is fitted on training data only, so the test set never influences selection,
tuning or scaling. Inside the leave one out loop the scaler is refitted on each
n-1 subset.

## Repository contents

| File | Purpose |
| --- | --- |
| `step1_model_screening.py` | Descriptor screening, hyperparameter search, test set scores  |
| `step2_loo_cv.py` | Leave-one-out-cross validation of the best model per property |
| `step3_validation.py` | Predictions on an independent validation workbook |
| `step4_shap_analysis.py` | SHAP values for the best model per property |
| `config/Final_Selected_Descriptors.csv` | Descriptors selected in the reported run, for reference |
| `requirements.txt` | Python dependencies |

## Installation

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

The code was tested with Python 3.12, numpy 2.4, pandas 3.0,
scikit learn 1.9, mlxtend 0.25, shap 0.52 and openpyxl 3.1.

## Input data

The datasets are not redistributed here. Each script
expects an `.xlsx` workbook with the following layout.

**Training workbook**: one sheet per property, named `Density`,
`Flash_Point`, `Freezing_Point` and `Viscosity`:

| Column | Content |
| --- | --- |
| `sample` (or `id`, `No`, `Index`) | Sample identifier, ignored as a feature |
| descriptor columns | Mole fraction weighted Mordred 2D descriptors, one column per descriptor |
| target column | Measured property; `density`, `flash`, `freez`, `viscosity`|

Columns that are entirely empty or entirely zero, or that contain any missing
value, are dropped automatically. Rows without a target value are dropped.

**Validation workbook**: one sheet per property, named either `Density`,
`Flash_Point`, `Freezing_Point`, `Viscosity` and one column per descriptor selected in step 1. Feature names must match the
step 1 selection exactly; the script reports missing or unexpected columns and
suggests near matches.

## Usage

Step 1, all four properties, seed 42:

```
python step1_model_screening.py --data /path/to/training_data.xlsx --outdir results
```

Useful options: `--seeds 42` ,
`--properties Density Flash_Point Freezing_Point Viscosity` , `--max-sfs 30`,
`--corr-threshold 0.9`, `--test-size 0.2`, `--cv 5`.

Steps 2 to 4 reuse `results/seed_42/Final_Selected_Descriptors.csv` by default:

```
python step2_loo_cv.py        --data /path/to/training_data.xlsx --outdir results
python step3_validation.py    --data /path/to/training_data.xlsx \
                              --validation /path/to/validation_data.xlsx --outdir results
python step4_shap_analysis.py --data /path/to/training_data.xlsx --outdir results
```

The best model and its hyperparameters are read from the `BEST_MODELS` block at
the top of steps 2 to 4. They hold the values reported in the manuscript; if
step 1 is rerun on other data, copy the new winners from
`model_performance_summary.xlsx` into that block.

## Outputs

| File | Content |
| --- | --- |
| `results/seed_<seed>/Table_S1_<Property>.csv` | Descriptor added at each SFS step with mean and standard deviation of the cross validated R2 |
| `results/seed_<seed>/Final_Selected_Descriptors.csv` | Final descriptor list per property, in selection order |
| `results/seed_<seed>/model_performance_summary.xlsx` | Train and test R2, RMSE and MAE for the six algorithms, with the selected hyperparameters |
| `results/seed_<seed>/predictions_detail.xlsx` | Measured and predicted values for every sample and model |
| `results/LOO_CV_summary.xlsx` | Leave one out R2, RMSE and MAE per property, plus per sample predictions |
| `results/validation_summary.xlsx` | Model, features and training scores used for the external validation, plus per sample predictions |
| `results/validation_predictions_all.xlsx` | Predictions of all four properties in one sheet |
| `results/shap/SHAP_<Property>_<Model>.xlsx` | Mean absolute SHAP per descriptor and per sample SHAP values |

## Runtime

Step 1 dominates the cost. With about 1800 candidate descriptors and
`--max-sfs 30` it takes several hours per property on four cores, because SFS
fits one random forest per candidate descriptor per step and fold.

Step 2 needs about 10 s per property for 300 samples. Step 4 takes roughly
5 to 15 min per property with 300 samples and 15 descriptors, because kernel
SHAP scales with the number of samples and with the background set.

## Notes and limitations

* SFS uses R2 with 5 fold cross validation on the training set; the best step is
  the one with the highest mean cross validated R2, not a fixed number of
  descriptors.
* SHAP values in step 4 are computed for all samples of the training workbook
  after the model has been fitted on the training split, matching the figures
  in the manuscript.
* Kernel SHAP is approximate for SVR; results depend slightly on the background
  sample, which is drawn with the fixed random seed.
* Single seed runs are reported in the manuscript. `--seeds` is provided for
  checking seed sensitivity.

## Data availability

The datasets compiled from the literature are not included in this repository.
They are available from the corresponding author on reasonable request.

## License

MIT, see `LICENSE`.

## Citation

If you use this code, please cite the article above and the archived release
(DOI to be added after the first Zenodo archive).

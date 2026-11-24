#!/usr/bin/env python3
"""
Step 1 - Descriptor screening and model benchmarking.

Pipeline
--------
column cleaning -> train/test split -> sequential forward selection (SFS)
-> Pearson de-correlation -> standardisation -> grid search -> test set scoring

The split is made before any selection so that the test set is never seen by
the feature selection or the hyperparameter search.

Usage
-----
    python step1_model_screening.py --data /path/to/training_data.xlsx --outdir results

The training workbook must contain one sheet per property (see README).
"""

import argparse
import os
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
PROPERTIES = {
    "Density": {
        "sheet": "Density",
        "target_keywords": ("\u5bc6\u5ea6", "density"),           # density
    },
    "Flash_Point": {
        "sheet": "Flash_Point",
        "target_keywords": ("\u95ea\u70b9", "flash"),             # flash point
    },
    "Freezing_Point": {
        "sheet": "Freezing_Point",
        "target_keywords": ("\u51b0\u70b9", "freez"),             # freezing point
    },
    "Viscosity": {
        "sheet": "Viscosity",
        "target_keywords": ("\u7c98\u5ea6", "viscosity"),         # viscosity
    },
}

# Sample identifier columns. The last three are the Chinese forms used
# by the authors' workbook: sample, number, index.
ID_COLUMNS = {"sample", "id", "name", "no", "index",
              "\u6837\u672c", "\u7f16\u53f7", "\u5e8f\u53f7"}

# Hyperparameter grids. One global set is used for all four properties; the
# best combination per property is reported in model_performance_summary.xlsx.
SVR_PARAMS = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 50, 100, 500],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
}

_kernel_grid = []
for ls in [0.5, 1.0, 5.0, 10.0, 50.0]:
    _kernel_grid.append(
        ConstantKernel(1.0)
        * RBF(length_scale=ls)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1.0))
    )
for nu in [0.5, 1.5]:
    for ls in [1.0, 5.0]:
        _kernel_grid.append(
            ConstantKernel(1.0)
            * Matern(length_scale=ls, nu=nu)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1.0))
        )
for ls in [1.0, 10.0]:
    _kernel_grid.append(
        ConstantKernel(1.0)
        * RationalQuadratic(length_scale=ls, alpha=1.0)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1.0))
    )
GPR_PARAMS = {"kernel": _kernel_grid, "alpha": [1e-4, 1e-3, 1e-2]}

ANN_PARAMS = {
    "hidden_layer_sizes": [
        (10,), (20,), (50,), (100,), (50, 25), (100, 50),
        (200,), (100, 50, 25), (128, 64, 32), (50, 50, 50),
    ],
    "activation": ["relu", "tanh", "logistic"],
    "alpha": [1e-5, 0.0001, 0.001, 0.01],
    "learning_rate_init": [0.0001, 0.001, 0.005, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [3000],
    "early_stopping": [True],
    "batch_size": ["auto", 32, 64],
}

RF_PARAMS = {
    "n_estimators": [100, 200, 300, 500, 800, 1000, 1500],
    "max_depth": [None, 3, 5, 7, 10, 15],
    "min_samples_split": [2, 3, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None, 0.5],
}

GBDT_PARAMS = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5, 10],
    "subsample": [0.6, 0.7, 0.8],
    "max_features": ["sqrt", 0.5],
}

MODEL_NAMES = ["MLR", "SVR", "GPR", "ANN", "RF", "GBDT"]

PARAM_MAP = {
    "SVR": SVR_PARAMS,
    "GPR": GPR_PARAMS,
    "ANN": ANN_PARAMS,
    "RF": RF_PARAMS,
    "GBDT": GBDT_PARAMS,
}


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Step 1: descriptor screening and model benchmarking."
    )
    p.add_argument("--data", required=True,
                   help="Training workbook (.xlsx), one sheet per property.")
    p.add_argument("--outdir", default="results",
                   help="Output directory (default: results).")
    p.add_argument("--properties", nargs="+", default=list(PROPERTIES),
                   choices=list(PROPERTIES),
                   help="Subset of properties to run (default: all four).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Random seeds; each seed runs the full pipeline "
                        "into its own sub directory (default: 42).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of samples held out as test set (default: 0.2).")
    p.add_argument("--cv", type=int, default=5,
                   help="Cross validation folds for SFS and grid search (default: 5).")
    p.add_argument("--max-sfs", type=int, default=30,
                   help="Maximum number of SFS forward steps (default: 30).")
    p.add_argument("--corr-threshold", type=float, default=0.9,
                   help="Pearson threshold for de-correlation (default: 0.9).")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel jobs for SFS and grid search (default: 4).")
    return p.parse_args()


def find_target_column(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if str(kw).lower() in str(col).lower():
                return col
    raise ValueError(f"No target column found (keywords: {keywords})")


def remove_allzero_and_allnan_columns(df, descriptor_cols):
    n_before = len(descriptor_cols)
    not_all_nan = [c for c in descriptor_cols if not df[c].isna().all()]
    not_all_zero = [c for c in not_all_nan if not (df[c] == 0).all()]
    print(f"Initial descriptor filter: {n_before} -> {len(not_all_zero)} columns")
    return not_all_zero


def remove_nan_columns(df, descriptor_cols):
    remaining = [c for c in descriptor_cols if not df[c].isna().any()]
    print(f"Drop columns containing NaN: removed "
          f"{len(descriptor_cols) - len(remaining)}, {len(remaining)} left")
    return remaining


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_base_estimator(name, seed):
    if name == "SVR":
        return SVR()
    if name == "GPR":
        return GaussianProcessRegressor(normalize_y=True, random_state=seed)
    if name == "ANN":
        return MLPRegressor(random_state=seed)
    if name == "RF":
        return RandomForestRegressor(random_state=seed)
    if name == "GBDT":
        return GradientBoostingRegressor(random_state=seed)
    return None


def pearson_correlation_filter(X_train, feature_names, threshold=0.9):
    """Drop later features that correlate above `threshold` with a kept one."""
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    n = len(feature_names)
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)
    keep_indices = [i for i in range(n) if i not in to_remove]
    print(f"Pearson de-correlation: {n} features in, {len(to_remove)} removed "
          f"(|r| > {threshold}), {len(keep_indices)} kept")
    return keep_indices


# ---------------------------------------------------------------------------
# 3. Pipeline for one property
# ---------------------------------------------------------------------------
def run_pipeline_for_property(prop_key, prop_info, seed, args):
    print(f"\n{'=' * 66}")
    print(f"Property: {prop_key} (sheet: {prop_info['sheet']}) [seed={seed}]")
    print("=" * 66)

    # ---- load and clean ---------------------------------------------------
    df = pd.read_excel(args.data, sheet_name=prop_info["sheet"])
    print(f"Raw data: {df.shape[0]} rows x {df.shape[1]} columns")
    target_col = find_target_column(df, prop_info["target_keywords"])
    print(f"Target column: {target_col}")

    exclude_cols = {target_col}
    exclude_cols.update(
        c for c in df.columns if str(c).strip().lower() in ID_COLUMNS
    )
    descriptor_cols = [c for c in df.columns if c not in exclude_cols]
    for c in descriptor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Candidate descriptors: {len(descriptor_cols)}")

    descriptor_cols = remove_allzero_and_allnan_columns(df, descriptor_cols)
    descriptor_cols = remove_nan_columns(df, descriptor_cols)
    if len(descriptor_cols) == 0:
        print("No usable descriptor left, property skipped.")
        return None, None, None, None

    df_clean = df[descriptor_cols + [target_col]].dropna(subset=[target_col])
    if len(df_clean) < 10:
        print("Fewer than 10 samples, property skipped.")
        return None, None, None, None

    X_full = df_clean[descriptor_cols].values.astype(float)
    y_full = df_clean[target_col].values.astype(float)
    ids_full = df_clean.index.values

    # ---- split first, so the test set stays untouched ---------------------
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X_full, y_full, ids_full, test_size=args.test_size, random_state=seed
    )
    print(f"Split: train {len(X_train)} samples, test {len(X_test)} samples")

    # ---- sequential forward selection on the training set only -----------
    print(f"Sequential forward selection on the training set "
          f"(max {args.max_sfs} steps) ...")
    sfs_start = time.time()
    estimator = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", random_state=seed, n_jobs=args.n_jobs
    )
    n_features = min(args.max_sfs, X_train.shape[1])
    sfs = SFS(
        estimator,
        k_features=n_features,
        forward=True,
        floating=False,
        scoring="r2",
        cv=args.cv,
        n_jobs=args.n_jobs,
    )
    sfs.fit(X_train, y_train)

    subsets = sfs.subsets_
    step_scores = {}
    for step, info in subsets.items():
        if "avg_score" in info:
            step_scores[step] = info["avg_score"]
        else:
            step_scores[step] = np.mean(info["cv_scores"])

    best_step = max(step_scores, key=step_scores.get)
    best_score = step_scores[best_step]
    best_indices = list(subsets[best_step]["feature_idx"])
    print(f"SFS finished in {time.time() - sfs_start:.1f}s")
    print(f"Best step: {best_step}, CV R2 = {best_score:.4f}, "
          f"{len(best_indices)} descriptors selected")

    # one newly added descriptor per step, as in SI Table S1
    history = []
    seen = set()
    for step in sorted(subsets.keys()):
        subset_info = subsets[step]
        feat_names = [descriptor_cols[i] for i in subset_info["feature_idx"]]
        cv_mean = subset_info.get("avg_score", np.mean(subset_info["cv_scores"]))
        cv_std = float(np.std(subset_info["cv_scores"])) if "cv_scores" in subset_info else 0.0
        for name in feat_names:
            if name not in seen:
                history.append((step, name, cv_mean, cv_std))
                seen.add(name)
                break

    # ---- Pearson de-correlation on the best subset ------------------------
    X_train_best = X_train[:, best_indices]
    best_feature_names = [descriptor_cols[i] for i in best_indices]
    keep_local_indices = pearson_correlation_filter(
        X_train_best, best_feature_names, threshold=args.corr_threshold
    )
    final_indices = [best_indices[i] for i in keep_local_indices]
    final_descs = [descriptor_cols[i] for i in final_indices]
    print(f"Descriptors passed to modelling: {len(final_descs)}")
    for i, d in enumerate(final_descs, 1):
        print(f"{i:2d}. {d}")

    X_train_sel = X_train[:, final_indices]
    X_test_sel = X_test[:, final_indices]

    # ---- standardisation (statistics from the training set only) ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    print("Standardisation done (mean/std from the training set)")

    # ---- train the six algorithms ----------------------------------------
    print("Training six models (grid search on the training set only)")
    summary_rows = []
    detail_rows = []
    model_start = time.time()

    for model_name in MODEL_NAMES:
        print(f"[{model_name}] training ...")
        t0 = time.time()
        if model_name == "MLR":
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            best_params = "N/A"
        else:
            base = build_base_estimator(model_name, seed)
            gs = GridSearchCV(
                base, PARAM_MAP[model_name], cv=args.cv, scoring="r2",
                n_jobs=args.n_jobs, verbose=0,
            )
            gs.fit(X_train_scaled, y_train)
            model = gs.best_estimator_
            best_params = str(gs.best_params_)
            print(f"Best parameters: {gs.best_params_}")

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        train_metrics = evaluate(y_train, y_train_pred)
        test_metrics = evaluate(y_test, y_test_pred)
        print(f"Train R2={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.4f}")
        print(f"Test  R2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}")

        summary_rows.append({
            "Property": prop_key,
            "Model": model_name,
            "BestParams": best_params,
            "Train_R2": round(train_metrics["R2"], 4),
            "Train_RMSE": round(train_metrics["RMSE"], 4),
            "Train_MAE": round(train_metrics["MAE"], 4),
            "Test_R2": round(test_metrics["R2"], 4),
            "Test_RMSE": round(test_metrics["RMSE"], 4),
            "Test_MAE": round(test_metrics["MAE"], 4),
        })

        for idx in range(len(y_train)):
            detail_rows.append({
                "SampleIndex": int(id_train[idx]),
                "Model": model_name,
                "Dataset": "train",
                "Measured": round(y_train[idx], 6),
                "Predicted": round(y_train_pred[idx], 6),
                "AbsError": round(abs(y_train[idx] - y_train_pred[idx]), 6),
            })
        for idx in range(len(y_test)):
            detail_rows.append({
                "SampleIndex": int(id_test[idx]),
                "Model": model_name,
                "Dataset": "test",
                "Measured": round(y_test[idx], 6),
                "Predicted": round(y_test_pred[idx], 6),
                "AbsError": round(abs(y_test[idx] - y_test_pred[idx]), 6),
            })
        print(f"Elapsed: {time.time() - t0:.1f}s")

    print(f"Modelling finished in {time.time() - model_start:.1f}s")
    detail_rows.sort(key=lambda x: x["SampleIndex"])
    return history, final_descs, summary_rows, detail_rows


# ---------------------------------------------------------------------------
# 4. Saving
# ---------------------------------------------------------------------------
def save_all_results(output_dir, all_selected, all_summary, all_details):
    try:
        if all_summary:
            pd.DataFrame(all_summary).to_excel(
                os.path.join(output_dir, "model_performance_summary.xlsx"), index=False
            )
        desc_rows = []
        for prop, desc_list in all_selected.items():
            for i, d in enumerate(desc_list, 1):
                desc_rows.append({"Property": prop, "Rank": i, "Descriptor": d})
        if desc_rows:
            pd.DataFrame(desc_rows).to_csv(
                os.path.join(output_dir, "Final_Selected_Descriptors.csv"), index=False
            )
        if all_details:
            with pd.ExcelWriter(
                os.path.join(output_dir, "predictions_detail.xlsx"), engine="openpyxl"
            ) as writer:
                for prop, details in all_details.items():
                    pd.DataFrame(details).to_excel(
                        writer, sheet_name=f"{prop}_detail", index=False
                    )
    except PermissionError as e:
        print(f"Could not write results (file open in Excel?): {e}")


def main():
    args = parse_args()
    print("=" * 70)
    print("column cleaning -> split -> SFS -> best step -> Pearson filter")
    print("-> standardisation -> grid search -> test set scoring")
    print(f"Random seeds: {args.seeds} (each seed gets its own sub directory)")
    print("=" * 70)

    os.makedirs(args.outdir, exist_ok=True)
    overall_start = time.time()

    for seed in args.seeds:
        seed_dir = os.path.join(args.outdir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n{'#' * 70}\n# seed={seed} -> {seed_dir}\n{'#' * 70}")

        all_history = {}
        all_selected = {}
        all_summary = []
        all_details = {}
        seed_start = time.time()

        for prop_key in args.properties:
            prop_info = PROPERTIES[prop_key]
            history, selected, summary, details = run_pipeline_for_property(
                prop_key, prop_info, seed, args
            )
            if summary is None:
                continue
            all_history[prop_key] = history
            all_selected[prop_key] = selected
            all_summary.extend(summary)
            all_details[prop_key] = details

            if history:
                df_sfs = pd.DataFrame(
                    history,
                    columns=["Step", "Added_Descriptor", "CV_R2_Mean", "CV_R2_Std"],
                )
                df_sfs.to_csv(
                    os.path.join(seed_dir, f"Table_S1_{prop_key}.csv"), index=False
                )
                df_sfs.to_excel(
                    os.path.join(seed_dir, f"Table_S1_{prop_key}.xlsx"), index=False
                )

            save_all_results(seed_dir, all_selected, all_summary, all_details)
            print(f"[seed={seed}] {prop_key} saved to {seed_dir}")
            elapsed = time.time() - seed_start
            done = args.properties.index(prop_key) + 1
            print(f"[seed={seed}] {done}/{len(args.properties)} done, "
                  f"elapsed {str(timedelta(seconds=int(elapsed)))}")

        if not all_summary:
            print(f"[seed={seed}] no property completed, nothing written")
        print(f"[seed={seed}] finished in "
              f"{str(timedelta(seconds=int(time.time() - seed_start)))}")

    print("\n" + "=" * 70)
    print(f"All seeds done in {str(timedelta(seconds=int(time.time() - overall_start)))}")
    print(f"Results: {args.outdir}/seed_<seed>/")


if __name__ == "__main__":
    main()

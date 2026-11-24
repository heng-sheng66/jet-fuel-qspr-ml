#!/usr/bin/env python3
"""
Step 2 - Leave one out cross validation of the best model.

For each property the best model and the hyperparameters found in step 1 are
refitted on every n-1 subset and the held out sample is predicted. Scaling is
refitted inside each fold, so no information leaks from the held out sample.

Usage
-----
    python step2_loo_cv.py --data /path/to/training_data.xlsx --outdir results
"""

import argparse
import os
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Best model per property, copied from step 1 (model_performance_summary.xlsx)
# Edit these entries if you rerun step 1 on your own data.
# ---------------------------------------------------------------------------
BEST_MODELS = {
    "Density": {
        "sheet": "Density",
        "target_keywords": ("\u5bc6\u5ea6", "density"),           # density
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 10, "epsilon": 0.001, "gamma": 0.01},
    },
    "Flash_Point": {
        "sheet": "Flash_Point",
        "target_keywords": ("\u95ea\u70b9", "flash"),             # flash point
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 500, "epsilon": 0.5, "gamma": "auto"},
    },
    "Freezing_Point": {
        "sheet": "Freezing_Point",
        "target_keywords": ("\u51b0\u70b9", "freez"),             # freezing point
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 500, "epsilon": 0.5, "gamma": "scale"},
    },
    "Viscosity": {
        "sheet": "Viscosity",
        "target_keywords": ("\u7c98\u5ea6", "viscosity"),         # viscosity
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 500, "epsilon": 0.001, "gamma": 0.01},
    },
}

# Sample identifier columns. The last three are the Chinese forms used
# by the authors' workbook: sample, number, index.
ID_COLUMNS = {"sample", "id", "name", "no", "index",
              "\u6837\u672c", "\u7f16\u53f7", "\u5e8f\u53f7"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 2: leave one out cross validation of the best model."
    )
    p.add_argument("--data", required=True, help="Training workbook (.xlsx).")
    p.add_argument("--outdir", default="results", help="Output directory.")
    p.add_argument("--selected", default=None,
                   help="Descriptor list from step 1 "
                        "(default: <outdir>/seed_42/Final_Selected_Descriptors.csv).")
    p.add_argument("--properties", nargs="+", default=list(BEST_MODELS),
                   choices=list(BEST_MODELS), help="Properties to run.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for stochastic models (default: 42).")
    p.add_argument("--outfile", default=None,
                   help="Output workbook (default: <outdir>/LOO_CV_summary.xlsx).")
    return p.parse_args()


def find_target_column(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if str(kw).lower() in str(col).lower():
                return col
    raise ValueError(f"No target column found (keywords: {keywords})")


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_model(model_name, params, seed):
    if model_name == "MLR":
        return LinearRegression(**params)
    if model_name == "SVR":
        return SVR(**params)
    if model_name == "GPR":
        return GaussianProcessRegressor(normalize_y=True, random_state=seed, **params)
    if model_name == "ANN":
        return MLPRegressor(random_state=seed, **params)
    if model_name == "RF":
        return RandomForestRegressor(random_state=seed, **params)
    if model_name == "GBDT":
        return GradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown model: {model_name}")


def load_selected_descriptors(path, prop_key):
    sel_df = pd.read_csv(path)
    descs = (sel_df[sel_df["Property"] == prop_key]
             .sort_values("Rank")["Descriptor"].tolist())
    if not descs:
        raise ValueError(f"[{prop_key}] no descriptor record in {path}")
    return descs


def run_loo_for_property(prop_key, cfg, args, selected_path):
    model_name = cfg["model"]
    best_params = cfg["params"]
    print(f"\n{'=' * 65}")
    print(f"Property: {prop_key}  |  best model: {model_name}")
    print("=" * 65)

    # ---- load and clean (same rules as step 1) ---------------------------
    df = pd.read_excel(args.data, sheet_name=cfg["sheet"])
    target_col = find_target_column(df, cfg["target_keywords"])
    print(f"Raw data: {df.shape[0]} rows, target column: {target_col}")

    exclude_cols = {target_col}
    exclude_cols.update(
        c for c in df.columns if str(c).strip().lower() in ID_COLUMNS
    )
    descriptor_cols = [c for c in df.columns if c not in exclude_cols]
    for c in descriptor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    not_all_nan = [c for c in descriptor_cols if not df[c].isna().all()]
    not_all_zero = [c for c in not_all_nan if not (df[c] == 0).all()]
    descriptor_cols = [c for c in not_all_zero if not df[c].isna().any()]
    df_clean = df[descriptor_cols + [target_col]].dropna(subset=[target_col])
    print(f"Clean data: {len(df_clean)} samples")

    final_descs = load_selected_descriptors(selected_path, prop_key)
    missing = [d for d in final_descs if d not in descriptor_cols]
    if missing:
        raise ValueError(f"[{prop_key}] training data is missing descriptors: {missing}")
    print(f"Descriptors from step 1: {len(final_descs)}")

    X = df_clean[final_descs].values.astype(float)
    y = df_clean[target_col].values.astype(float).ravel()
    orig_ids = df_clean.index.values
    n_samples = len(X)
    print(f"Samples: {n_samples}, features: {X.shape[1]}")
    print(f"Leave one out: {n_samples} folds")

    # ---- leave one out ---------------------------------------------------
    loo = LeaveOneOut()
    y_pred = np.zeros(n_samples)
    loo_start = time.time()
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        if (fold + 1) % 20 == 0 or fold == 0 or fold == n_samples - 1:
            elapsed = time.time() - loo_start
            avg_time = elapsed / (fold + 1)
            remaining = avg_time * (n_samples - fold - 1)
            print(f"Fold {fold + 1}/{n_samples}  elapsed {elapsed:.1f}s  "
                  f"eta {remaining:.0f}s")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = build_model(model_name, best_params, args.seed)
        model.fit(X_train_scaled, y_train)
        y_pred[fold] = model.predict(X_test_scaled)[0]
    print(f"LOO finished in {time.time() - loo_start:.1f}s")

    metrics = evaluate(y, y_pred)
    print("\nLOO metrics:")
    print(f"R2   = {metrics['R2']:.4f}")
    print(f"RMSE = {metrics['RMSE']:.4f}")
    print(f"MAE  = {metrics['MAE']:.4f}")

    detail_rows = []
    for i in range(n_samples):
        detail_rows.append({
            "SampleIndex": int(orig_ids[i]),
            "Measured": round(y[i], 6),
            "Predicted": round(y_pred[i], 6),
            "AbsError": round(abs(y[i] - y_pred[i]), 6),
        })
    return metrics, detail_rows, final_descs


def main():
    args = parse_args()
    if args.selected is None:
        args.selected = os.path.join(args.outdir, "seed_42",
                                     "Final_Selected_Descriptors.csv")
    if args.outfile is None:
        args.outfile = os.path.join(args.outdir, "LOO_CV_summary.xlsx")

    print("=" * 65)
    print("Leave one out cross validation of the best models from step 1")
    print("descriptors come from: " + args.selected)
    print("=" * 65)

    overall_start = time.time()
    all_summary = []
    property_details = {}
    failed = []

    for prop_key in args.properties:
        cfg = BEST_MODELS[prop_key]
        idx = args.properties.index(prop_key)
        prop_start = time.time()
        try:
            metrics, detail, final_descs = run_loo_for_property(
                prop_key, cfg, args, args.selected
            )
        except (ValueError, KeyError) as err:
            print(f"\n[skipped] {prop_key}: {err}")
            failed.append({"Property": prop_key,
                           "Reason": str(err).replace("\n", " | ")})
            continue

        property_details[prop_key] = detail
        all_summary.append({
            "Property": prop_key,
            "Model": cfg["model"],
            "BestParams": str(cfg["params"]),
            "NFeatures": len(final_descs),
            "LOO_R2": round(metrics["R2"], 4),
            "LOO_RMSE": round(metrics["RMSE"], 4),
            "LOO_MAE": round(metrics["MAE"], 4),
            "NSamples": len(detail),
        })
        elapsed = time.time() - overall_start
        print(f"\n[ok] {prop_key} ({idx + 1}/{len(args.properties)}), "
              f"{time.time() - prop_start:.1f}s, "
              f"total {str(timedelta(seconds=int(elapsed)))}")

    print("\n" + "=" * 65)
    print(f"Writing {args.outfile}")
    with pd.ExcelWriter(args.outfile, engine="openpyxl") as writer:
        pd.DataFrame(all_summary).to_excel(
            writer, sheet_name="LOO_summary", index=False
        )
        if failed:
            pd.DataFrame(failed).to_excel(
                writer, sheet_name="skipped", index=False
            )
        for prop_key, detail in property_details.items():
            pd.DataFrame(detail).to_excel(
                writer, sheet_name=f"{prop_key}_LOO", index=False
            )

    print("\n" + "=" * 78)
    print(f"{'Property':<16}{'Model':<8}{'NFeat':<8}{'LOO_R2':<10}"
          f"{'LOO_RMSE':<12}{'LOO_MAE':<12}")
    print("-" * 78)
    for r in all_summary:
        print(f"{r['Property']:<16}{r['Model']:<8}{r['NFeatures']:<8}"
              f"{r['LOO_R2']:<10.4f}{r['LOO_RMSE']:<12.4f}{r['LOO_MAE']:<12.4f}")
    print("=" * 78)
    for f in failed:
        print(f"- {f['Property']}: {f['Reason']}")
    print(f"\n[done] {len(all_summary)} properties, "
          f"total {str(timedelta(seconds=int(time.time() - overall_start)))}")


if __name__ == "__main__":
    main()

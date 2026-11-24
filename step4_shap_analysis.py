#!/usr/bin/env python3
"""
Step 4 - SHAP analysis of the best model.

The model selected in step 1 is refitted with the hyperparameters reported
there and SHAP values are computed for every sample of the training workbook.
Support vector regression is explained with shap.KernelExplainer using a
random background set drawn from the training data; tree models use the exact
tree explainer.

Usage
-----
    python step4_shap_analysis.py --data /path/to/training_data.xlsx --outdir results
"""

import argparse
import os
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
        description="Step 4: SHAP analysis of the best model."
    )
    p.add_argument("--data", required=True, help="Training workbook (.xlsx).")
    p.add_argument("--outdir", default="results", help="Output directory.")
    p.add_argument("--selected", default=None,
                   help="Descriptor list from step 1 "
                        "(default: <outdir>/seed_42/Final_Selected_Descriptors.csv).")
    p.add_argument("--shap-outdir", default=None,
                   help="SHAP output directory (default: <outdir>/shap).")
    p.add_argument("--properties", nargs="+", default=list(BEST_MODELS),
                   choices=list(BEST_MODELS), help="Properties to run.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test fraction used in step 1 (default: 0.2).")
    p.add_argument("--background", type=int, default=100,
                   help="Background samples for KernelExplainer (default: 100).")
    return p.parse_args()


def find_target_column(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if str(kw).lower() in str(col).lower():
                return col
    raise ValueError(f"No target column found (keywords: {keywords})")


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


def compute_shap_values(model_name, model, X_explain, X_train_scaled, seed, n_background):
    np.random.seed(seed)
    if model_name in ("RF", "GBDT"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain)
    else:
        background = shap.sample(
            X_train_scaled,
            min(n_background, X_train_scaled.shape[0]),
            random_state=seed,
        )
        explainer = shap.KernelExplainer(lambda X: model.predict(X), background)
        sv = explainer.shap_values(X_explain)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[0]
    return sv


def run_shap_for_property(prop_key, cfg, args, selected_path):
    model_name = cfg["model"]
    print(f"\n{'=' * 65}")
    print(f"Property: {prop_key}  |  best model: {model_name}")
    print("=" * 65)

    # ---- load and clean --------------------------------------------------
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
    print(f"Clean data: {len(df_clean)} samples, {len(descriptor_cols)} descriptors")

    final_descs = load_selected_descriptors(selected_path, prop_key)
    missing = [d for d in final_descs if d not in descriptor_cols]
    if missing:
        raise ValueError(f"[{prop_key}] training data is missing descriptors: {missing}")
    print(f"Descriptors from step 1: {len(final_descs)}")

    X_full = df_clean[final_descs].values.astype(float)
    y_full = df_clean[target_col].values.astype(float).ravel()

    # ---- same split and scaling as step 1 -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=args.test_size, random_state=args.seed
    )
    print(f"Split: train {len(X_train)}, test {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("Standardisation done (mean/std from the training set only)")

    print(f"Fitting {model_name} with the hyperparameters from step 1 ...")
    model = build_model(model_name, cfg["params"], args.seed)
    model.fit(X_train_scaled, y_train)
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(scaler.transform(X_test), y_test)
    print(f"Check: train R2={train_r2:.4f}, test R2={test_r2:.4f}")

    # ---- SHAP for all samples -------------------------------------------
    X_all_scaled = scaler.transform(X_full)
    print(f"Computing SHAP values ({X_all_scaled.shape[0]} samples x "
          f"{X_all_scaled.shape[1]} features) ...")
    t0 = time.time()
    shap_values = compute_shap_values(
        model_name, model, X_all_scaled, X_train_scaled, args.seed, args.background
    )
    print(f"SHAP finished in {time.time() - t0:.1f}s")

    # ---- summary --------------------------------------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    summary_rows = []
    for rank, name in enumerate(final_descs, 1):
        summary_rows.append({
            "Rank": rank,
            "Feature": name,
            "MeanAbsSHAP": round(float(mean_abs[rank - 1]), 6),
        })

    # ---- per sample detail ----------------------------------------------
    detail_rows = []
    for i in range(X_all_scaled.shape[0]):
        for rank, name in enumerate(final_descs, 1):
            detail_rows.append({
                "SampleIndex": i + 1,
                "Feature": name,
                "FeatureRank": rank,
                "Value_scaled": round(float(X_all_scaled[i, rank - 1]), 4),
                "Value_raw": round(float(X_full[i, rank - 1]), 4),
                "SHAP": round(float(shap_values[i, rank - 1]), 6),
            })

    out_file = os.path.join(args.shap_outdir, f"SHAP_{prop_key}_{model_name}.xlsx")
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="SHAP_summary", index=False)
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="SHAP_detail", index=False)
    print(f"Written: {out_file}")

    return {
        "Property": prop_key,
        "Model": model_name,
        "NFeatures": len(final_descs),
        "NSamples": X_all_scaled.shape[0],
        "SHAP_seconds": round(time.time() - t0, 1),
    }, summary_rows


def main():
    args = parse_args()
    if args.selected is None:
        args.selected = os.path.join(args.outdir, "seed_42",
                                     "Final_Selected_Descriptors.csv")
    if args.shap_outdir is None:
        args.shap_outdir = os.path.join(args.outdir, "shap")
    os.makedirs(args.shap_outdir, exist_ok=True)

    print("=" * 65)
    print("SHAP analysis of the best models from step 1")
    print("descriptors come from: " + args.selected)
    print("=" * 65)
    overall_start = time.time()
    all_brief = []

    for prop_key in args.properties:
        cfg = BEST_MODELS[prop_key]
        idx = args.properties.index(prop_key)
        try:
            brief, summary_rows = run_shap_for_property(
                prop_key, cfg, args, args.selected
            )
        except (ValueError, KeyError) as err:
            print(f"\n[skipped] {prop_key}: {err}")
            continue

        all_brief.append(brief)
        top = sorted(summary_rows, key=lambda r: -r["MeanAbsSHAP"])[:5]
        print("Top five features by mean absolute SHAP:")
        for r in top:
            print(f"  {r['Feature']:<20} {r['MeanAbsSHAP']:.6f}")

        elapsed = time.time() - overall_start
        print(f"\n[ok] {prop_key} ({idx + 1}/{len(args.properties)}), total "
              f"{str(timedelta(seconds=int(elapsed)))}")

    print("\n" + "=" * 78)
    print(f"{'Property':<16}{'Model':<8}{'NFeat':<8}{'NSamples':<10}{'SHAP_s':<12}")
    print("-" * 78)
    for r in all_brief:
        print(f"{r['Property']:<16}{r['Model']:<8}{r['NFeatures']:<8}"
              f"{r['NSamples']:<10}{r['SHAP_seconds']:<12}")
    print("=" * 78)
    print(f"\nResults: {args.shap_outdir}")
    print(f"[done] total {str(timedelta(seconds=int(time.time() - overall_start)))}")


if __name__ == "__main__":
    main()

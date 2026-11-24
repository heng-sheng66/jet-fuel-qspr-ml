#!/usr/bin/env python3
"""
Step 3 - External validation set predictions.

The best model from step 1 is refitted on the training workbook with the
hyperparameters reported there and used to predict an independent validation
workbook. The validation features must match the descriptors selected in
step 1 exactly; the script stops with a readable message otherwise.

Usage
-----
    python step3_validation.py --data training.xlsx --validation validation.xlsx \
        --outdir results
"""

import argparse
import difflib
import os
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 10, "epsilon": 0.001, "gamma": 0.01},
    },
    "Flash_Point": {
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 500, "epsilon": 0.5, "gamma": "auto"},
    },
    "Freezing_Point": {
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 500, "epsilon": 0.5, "gamma": "scale"},
    },
    "Viscosity": {
        "model": "SVR",
        "params": {"kernel": "rbf", "C": 500, "epsilon": 0.001, "gamma": 0.01},
    },
}

PROPERTIES = {
    "Density": {
        "sheet": "Density",
        "val_sheet": ("\u5bc6\u5ea6", "Density"),
        "target_keywords": ("\u5bc6\u5ea6", "density"),           # density
    },
    "Flash_Point": {
        "sheet": "Flash_Point",
        "val_sheet": ("\u95ea\u70b9", "Flash_Point"),
        "target_keywords": ("\u95ea\u70b9", "flash"),             # flash point
    },
    "Freezing_Point": {
        "sheet": "Freezing_Point",
        "val_sheet": ("\u51b0\u70b9", "Freezing_Point"),
        "target_keywords": ("\u51b0\u70b9", "freez"),             # freezing point
    },
    "Viscosity": {
        "sheet": "Viscosity",
        "val_sheet": ("\u7c98\u5ea6", "Viscosity"),
        "target_keywords": ("\u7c98\u5ea6", "viscosity"),         # viscosity
    },
}

# Sample identifier columns. The last three are the Chinese forms used
# by the authors' workbook: sample, number, index.
ID_COLUMNS = {"sample", "id", "name", "no", "index",
              "\u6837\u672c", "\u7f16\u53f7", "\u5e8f\u53f7"}
VAL_ID_CANDIDATES = ("\u5e8f\u53f7", "sample", "Sample", "ID", "id",
                     "\u7f16\u53f7", "No")
VAL_NON_FEATURE_COLS = {"\u5e8f\u53f7", "\u6837\u54c1\u7f16\u53f7",
                        "\u6837\u672c\u7f16\u53f7", "\u7f16\u53f7",
                        "\u6837\u672c", "sample", "id", "name"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 3: predict an independent validation set."
    )
    p.add_argument("--data", required=True, help="Training workbook (.xlsx).")
    p.add_argument("--validation", required=True, help="Validation workbook (.xlsx).")
    p.add_argument("--outdir", default="results", help="Output directory.")
    p.add_argument("--selected", default=None,
                   help="Descriptor list from step 1 "
                        "(default: <outdir>/seed_42/Final_Selected_Descriptors.csv).")
    p.add_argument("--properties", nargs="+", default=list(PROPERTIES),
                   choices=list(PROPERTIES), help="Properties to run.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the train/test split (default: 42).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test fraction used in step 1 (default: 0.2).")
    return p.parse_args()


def find_target_column(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if str(kw).lower() in str(col).lower():
                return col
    raise ValueError(f"No target column found (keywords: {keywords})")


def pick_sheet(names, candidates):
    for cand in candidates:
        if cand in names:
            return cand
    return None


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


def run_validation_for_property(prop_key, prop_info, args, selected_path):
    model_info = BEST_MODELS[prop_key]
    model_name = model_info["model"]
    best_params = model_info["params"]
    print(f"\n{'=' * 65}")
    print(f"Property: {prop_key}  |  best model: {model_name}")
    print("=" * 65)

    # ---- training data ---------------------------------------------------
    df = pd.read_excel(args.data, sheet_name=prop_info["sheet"])
    target_col = find_target_column(df, prop_info["target_keywords"])
    exclude_cols = {target_col}
    exclude_cols.update(
        c for c in df.columns if str(c).strip().lower() in ID_COLUMNS
    )
    descriptor_cols = [c for c in df.columns if c not in exclude_cols]
    for c in descriptor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Raw data: {df.shape[0]} rows x {len(descriptor_cols)} descriptors")
    print(f"Target column: {target_col}")

    not_all_nan = [c for c in descriptor_cols if not df[c].isna().all()]
    not_all_zero = [c for c in not_all_nan if not (df[c] == 0).all()]
    descriptor_cols = [c for c in not_all_zero if not df[c].isna().any()]
    df_clean = df[descriptor_cols + [target_col]].dropna(subset=[target_col])
    X_full = df_clean[descriptor_cols].values.astype(float)
    y_full = df_clean[target_col].values.astype(float)
    print(f"Clean data: {len(X_full)} samples, {len(descriptor_cols)} descriptors")

    # ---- same split as step 1 -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=args.test_size, random_state=args.seed
    )
    print(f"Split: train {len(X_train)}, test {len(X_test)}")

    # ---- descriptors selected in step 1 ---------------------------------
    sel_df = pd.read_csv(selected_path)
    model_descs = (sel_df[sel_df["Property"] == prop_key]
                   .sort_values("Rank")["Descriptor"].tolist())
    if not model_descs:
        raise ValueError(f"[{prop_key}] no descriptor record in {selected_path}")
    model_short = [d.replace("mordred_", "") for d in model_descs]
    print(f"Descriptors from step 1: {len(model_descs)}")

    # ---- validation workbook --------------------------------------------
    val_sheet = pick_sheet(pd.ExcelFile(args.validation).sheet_names,
                           prop_info["val_sheet"])
    if val_sheet is None:
        available = pd.ExcelFile(args.validation).sheet_names
        raise ValueError(
            f"[{prop_key}] validation sheet not found "
            f"(tried {prop_info['val_sheet']}). Available sheets: {available}"
        )
    val_df = pd.read_excel(args.validation, sheet_name=val_sheet)

    val_id_col = next((c for c in VAL_ID_CANDIDATES if c in val_df.columns), None)
    if val_id_col is None:
        raise ValueError(
            f"[{prop_key}] no sample id column in validation sheet "
            f"'{val_sheet}'. Columns found: {list(val_df.columns)[:10]} ..."
        )
    val_feature_names = [c for c in val_df.columns
                         if str(c).strip() not in VAL_NON_FEATURE_COLS]
    print(f"Validation sheet '{val_sheet}': {val_df.shape[0]} samples, "
          f"{len(val_feature_names)} features")

    # ---- features must match exactly ------------------------------------
    model_set = set(model_short)
    val_set = set(val_feature_names)
    missing_in_val = [f for f in model_short if f not in val_set]
    extra_in_val = [f for f in val_feature_names if f not in model_set]
    if missing_in_val or extra_in_val:
        msg = [f"[{prop_key}] validation features do not match the model:"]
        if missing_in_val:
            msg.append(f"  missing descriptors: {missing_in_val}")
        if extra_in_val:
            msg.append(f"  unexpected descriptors: {extra_in_val}")
        for f in missing_in_val:
            close = difflib.get_close_matches(f, val_feature_names, n=3, cutoff=0.7)
            if close:
                msg.append(f"  hint: is {close} a typo of '{f}'?")
        raise ValueError("\n".join(msg))
    print("Feature check passed: validation features match the model")

    # ---- keep the order used in step 1 ----------------------------------
    train_short2full = {c.replace("mordred_", ""): c for c in descriptor_cols}
    selected_train_cols = [train_short2full[s] for s in model_short]
    for i, d in enumerate(selected_train_cols, 1):
        print(f"{i:2d}. {d}")
    feat_idx = [descriptor_cols.index(c) for c in selected_train_cols]
    X_train_sel = X_train[:, feat_idx]
    X_test_sel = X_test[:, feat_idx]
    val_X = val_df[model_short].values.astype(float)

    # ---- standardisation ------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    print("Standardisation done (mean/std from the training set only)")

    # ---- refit and score ------------------------------------------------
    print(f"\nFitting {model_name} with the hyperparameters from step 1")
    print(f"Parameters: {best_params}")
    model = build_model(model_name, best_params, args.seed)
    model.fit(X_train_scaled, y_train)

    train_m = evaluate(y_train, model.predict(X_train_scaled))
    test_m = evaluate(y_test, model.predict(X_test_scaled))
    print(f"Train R2={train_m['R2']:.4f}  RMSE={train_m['RMSE']:.4f}  "
          f"MAE={train_m['MAE']:.4f}")
    print(f"Test  R2={test_m['R2']:.4f}  RMSE={test_m['RMSE']:.4f}  "
          f"MAE={test_m['MAE']:.4f}")

    # ---- missing values in the validation set ---------------------------
    train_feature_means = X_train_sel.mean(axis=0)
    nan_mask = np.isnan(val_X)
    if nan_mask.any():
        print("\nValidation set contains missing values, filled with the "
              "training set mean:")
        for r, c in np.argwhere(nan_mask):
            sample_id = val_df.iloc[r][val_id_col]
            print(f"  sample {sample_id}, '{model_short[c]}' -> "
                  f"{train_feature_means[c]:.6g}")
        val_X = np.where(nan_mask, train_feature_means, val_X)
    else:
        print("Validation set has no missing values")

    X_val_scaled = scaler.transform(val_X)
    val_preds = model.predict(X_val_scaled)
    print("\nPredictions on the validation set:")
    for i in range(len(val_preds)):
        print(f"  sample {val_df.iloc[i][val_id_col]}: {val_preds[i]:.4f}")

    # ---- save per property ----------------------------------------------
    pred_rows = []
    for i in range(len(val_preds)):
        row = {
            "SampleID": val_df.iloc[i][val_id_col],
            "Predicted": round(float(val_preds[i]), 6),
        }
        for j, fname in enumerate(model_short):
            row[fname] = float(val_X[i][j])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(args.outdir, f"validation_predictions_{prop_key}.csv"),
        index=False,
    )

    result = {
        "Property": prop_key,
        "BestModel": model_name,
        "BestParams": str(best_params),
        "NFeatures": len(selected_train_cols),
        "Features": ", ".join(selected_train_cols),
        "Train_R2": round(train_m["R2"], 4),
        "Train_RMSE": round(train_m["RMSE"], 4),
        "Train_MAE": round(train_m["MAE"], 4),
        "Test_R2": round(test_m["R2"], 4),
        "Test_RMSE": round(test_m["RMSE"], 4),
        "Test_MAE": round(test_m["MAE"], 4),
    }
    return result, val_preds, val_df, val_id_col


def main():
    args = parse_args()
    if args.selected is None:
        args.selected = os.path.join(args.outdir, "seed_42",
                                     "Final_Selected_Descriptors.csv")
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 65)
    print("External validation of the best models from step 1")
    print("=" * 65)
    overall_start = time.time()
    all_results = []
    all_val_rows = []
    failed = []

    for prop_key in args.properties:
        idx = args.properties.index(prop_key)
        try:
            result, val_preds, val_df, val_id_col = run_validation_for_property(
                prop_key, PROPERTIES[prop_key], args, args.selected
            )
        except ValueError as err:
            print(f"\n[skipped] {prop_key}: {err}")
            failed.append({"Property": prop_key,
                           "Reason": str(err).replace("\n", " | ")})
            continue

        all_results.append(result)
        for i in range(len(val_preds)):
            all_val_rows.append({
                "Property": prop_key,
                "SampleID": val_df.iloc[i][val_id_col],
                "BestModel": result["BestModel"],
                "Predicted": round(float(val_preds[i]), 6),
            })
        print(f"\n[ok] {prop_key} ({idx + 1}/{len(args.properties)}), total "
              f"{str(timedelta(seconds=int(time.time() - overall_start)))}")

    if all_results:
        summary_path = os.path.join(args.outdir, "validation_summary.xlsx")
        pd.DataFrame(all_results).to_excel(summary_path, index=False)
        print(f"\nSummary written to {summary_path}")
    else:
        print("\nNo property completed, nothing written")

    if all_val_rows:
        wide_path = os.path.join(args.outdir, "validation_predictions_all.xlsx")
        wide_df = (pd.DataFrame(all_val_rows)
                   .pivot(index="SampleID", columns="Property", values="Predicted")
                   .reset_index())
        wide_df.to_excel(wide_path, index=False)
        print(f"All predictions written to {wide_path}")

    print("\n" + "=" * 78)
    print(f"{'Property':<16}{'Model':<8}{'NFeat':<8}{'Test_R2':<10}"
          f"{'Test_RMSE':<12}{'Prediction range':<20}")
    print("-" * 78)
    for r in all_results:
        preds = [v["Predicted"] for v in all_val_rows if v["Property"] == r["Property"]]
        pred_range = f"{min(preds):.2f} ~ {max(preds):.2f}"
        print(f"{r['Property']:<16}{r['BestModel']:<8}{r['NFeatures']:<8}"
              f"{r['Test_R2']:<10.4f}{r['Test_RMSE']:<12.4f}{pred_range:<20}")
    print("=" * 78)

    if failed:
        print("\nProperties that could not be completed:")
        for f in failed:
            print(f"- {f['Property']}: {f['Reason']}")
    print(f"\n[done] {len(all_results)} properties, total "
          f"{str(timedelta(seconds=int(time.time() - overall_start)))}")


if __name__ == "__main__":
    main()

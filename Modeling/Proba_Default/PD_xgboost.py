from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path("/Users/yunchan/Desktop/PD_model_v3/dataset/dataset_v5.1.csv")
ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "xgboost_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CONFUSION_MATRIX_CSV = ARTIFACT_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG = ARTIFACT_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = ARTIFACT_DIR / "roc_curve.png"
PR_CURVE_PNG = ARTIFACT_DIR / "pr_curve.png"
FEATURE_IMPORTANCE_PNG = ARTIFACT_DIR / "feature_importance.png"
SHAP_SUMMARY_CSV = ARTIFACT_DIR / "shap_summary.csv"
SHAP_SUMMARY_PNG = ARTIFACT_DIR / "shap_summary.png"
SHAP_SAMPLE_SIZE = 5000


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target"] = np.where(df["loan_status"].eq("Charged Off"), 1, 0)
    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    annual_income = df["monthly_inc"] * 12
    annual_income = annual_income.replace(0, 1)
    df["dti_ratio"] = (df["loan_amnt"] / annual_income).replace([np.inf, -np.inf], 0)

    card_limit_safe = df["card_limit"].replace(0, 1)
    df["credit_utilization"] = (df["monthly_card_usage"] / card_limit_safe).clip(0, 1)
    df["avg_card_limit"] = (df["card_limit"] / df["card_count"].replace(0, 1)).replace(
        [np.inf, -np.inf], 0
    )

    monthly_inc_safe = df["monthly_inc"].replace(0, 1)
    df["card_usage_to_income"] = (df["monthly_card_usage"] / monthly_inc_safe).replace(
        [np.inf, -np.inf], 0
    )

    df["employment_years_group"] = pd.cut(
        df["employment_years"],
        bins=[-0.1, 1, 3, 5, 10, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)

    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_numeric"] = df["grade"].map(grade_map)

    income_level_cat, _ = pd.qcut(
        df["monthly_inc"],
        q=3,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    df["income_level"] = pd.Series(income_level_cat).astype("Int64").fillna(0).astype(int)

    df["grade_income_interaction"] = df["grade_numeric"] * (df["income_level"] + 1)
    df["dti_credit_interaction"] = df["dti_ratio"] * df["credit_utilization"]

    df["log_monthly_inc"] = np.log1p(df["monthly_inc"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])

    df["loan_to_income_monthly"] = safe_divide(df["loan_amnt"], monthly_inc_safe)
    df["card_limit_to_income"] = safe_divide(card_limit_safe, monthly_inc_safe)
    df["credit_remaining_ratio"] = (card_limit_safe - df["monthly_card_usage"]) / card_limit_safe
    df["credit_remaining_ratio"] = df["credit_remaining_ratio"].clip(0, 1)
    df["debt_to_card_limit"] = safe_divide(df["loan_amnt"], card_limit_safe)

    df["employment_years_log"] = np.log1p(df["employment_years"].clip(lower=0))
    df["delinq_2yrs_flag"] = (df["delinq_2yrs"] >= 2).astype(int)
    df["grade_employment_interaction"] = df["grade_numeric"] * (df["employment_years_log"] + 1)

    df = df.drop(columns=["loan_status", "grade"])
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


def train_model(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    conf_mat = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(conf_mat, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    conf_df.to_csv(CONFUSION_MATRIX_CSV)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nArtifacts saved to:", ARTIFACT_DIR)

    return metrics


def plot_confusion_matrix(conf_mat: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(conf_mat, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), value in np.ndenumerate(conf_mat):
        ax.text(j, i, f"{value}", ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close(fig)


def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=300)
    plt.close(fig)


def plot_pr_curve(y_test: pd.Series, y_proba: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR Curve (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PR_CURVE_PNG, dpi=300)
    plt.close(fig)


def plot_feature_importance(model: XGBClassifier, feature_names: list[str]) -> None:
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="gain")
    # Ensure all features represented
    importances = pd.Series(importance_dict)
    importances.index.name = "feature"
    importances.name = "gain"
    importances = importances.reindex(feature_names, fill_value=0)
    top_features = importances.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_features.sort_values().plot(kind="barh", ax=ax, color="skyblue")
    ax.set_xlabel("Gain Importance")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PNG, dpi=300)
    plt.close(fig)


def generate_shap_artifacts(model: XGBClassifier, X: pd.DataFrame) -> None:
    if len(X) > SHAP_SAMPLE_SIZE:
        X_sample = X.sample(SHAP_SAMPLE_SIZE, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    summary_df = pd.DataFrame(
        {"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap}
    ).sort_values(by="mean_abs_shap", ascending=False)
    summary_df.to_csv(SHAP_SUMMARY_CSV, index=False)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_dataset(DATA_PATH)
    df = engineer_features(df)

    feature_columns = [col for col in df.columns if col != "target"]
    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = train_model(X_train, y_train, scale_pos_weight)
    print("Model training complete.")

    metrics = evaluate_model(model, X_test, y_test)
    conf_mat = confusion_matrix(y_test, model.predict(X_test))
    y_proba = model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(conf_mat)
    plot_roc_curve(y_test, y_proba)
    plot_pr_curve(y_test, y_proba)
    plot_feature_importance(model, feature_columns)
    generate_shap_artifacts(model, X_test)

    joblib.dump({"model": model, "feature_names": feature_columns}, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


if __name__ == "__main__":
    main()

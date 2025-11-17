# train_classifier_lightgbm_tuned.py
"""
LightGBM 분류 모델을 GridSearchCV로 튜닝 후 최적 모델을 저장하는 코드
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

import sys
# data_utils.py 경로
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


DATA_PATH = "../../Data_Madang/auction_preprocessed.csv"
RANDOM_STATE = 42
MODEL_OUT = "../trained_model/auction_classifier_lgbm_tuned.joblib"


# ------------------------------------------------
# 모델 평가 함수
# ------------------------------------------------
def evaluate_model(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0)
    }


# ------------------------------------------------
# 메인 함수
# ------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, DATA_PATH)

    # 1. 데이터 로드 및 정제
    df = data_utils.load_and_clean(full_data_path)
    if df is None:
        return

    # 절차적 종료 제외 → 실제 낙찰된 건만 사용
    df = df[df['낙찰가'].notnull()].copy()

    # 라벨 생성 및 증강 및 피처엔지니어링
    df = data_utils.define_label(df)
    df = data_utils.augment_data(df)
    df = data_utils.feature_engineer(df)

    # 2. 학습 데이터 준비
    X, y, num_imputer, scaler = data_utils.prepare_training_data(df)

    # 3. Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] 데이터 분할 완료: train {len(X_train)} / test {len(X_test)}")

    # ------------------------------------------------
    # 4. 하이퍼파라미터 후보 (GridSearch)
    # ------------------------------------------------
    param_grid = {
        'n_estimators': [300, 500],
        'learning_rate': [0.01, 0.05],
        'max_depth': [-1, 6, 10],
        'num_leaves': [31, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # ------------------------------------------------
    # 5. GridSearchCV 진행 (AUC 기준)
    # ------------------------------------------------
    print("\n[INFO] GridSearchCV 시작...")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\n[INFO] GridSearch 완료")
    print(f"[INFO] Best AUC: {grid.best_score_:.4f}")
    print(f"[INFO] Best Params: {grid.best_params_}")

    best_model = grid.best_estimator_

    # ------------------------------------------------
    # 6. Test Set 평가
    # ------------------------------------------------
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print("\n--- 최종 모델 평가 결과 (Test Set) ---")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # ------------------------------------------------
    # 7. 모델 저장
    # ------------------------------------------------
    model_path = os.path.join(base_dir, MODEL_OUT)
    model_pack = {
        "model": best_model,
        "imputer": num_imputer,
        "scaler": scaler,
        "features": list(X.columns),
        "best_params": grid.best_params_
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_pack, model_path)

    print(f"\n[INFO] 최적 모델 저장 완료: {model_path}")


if __name__ == "__main__":
    main()

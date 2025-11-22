# train.py
"""
데이터를 불러와 전처리 및 증강을 수행하고, 분류 모델을 학습시킨 후 저장합니다.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

import sys
import os
# data_utils.py 경로 설정을 위해 상위 폴더를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data_utils.py 에서 모든 전처리 함수들을 가져옵니다.
import data_utils

DATA_PATH = "../../Data_Madang/auction_preprocessed.csv"
RANDOM_STATE = 42
# MODEL_OUT = "../trained_model/auction_classifier_xgb.joblib"

# ---------------------------
# 2) 모델 학습 및 평가 함수
# ---------------------------
def train_xgb(X_train, y_train, X_val, y_val, params=None, num_round=300):
    """XGBoost 분류 모델을 학습시킵니다."""
    if params is None:
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "eta": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": RANDOM_STATE
        }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(
        params, dtrain, num_boost_round=num_round,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=50, early_stopping_rounds=20
    )
    return bst

def evaluate_model(bst, X, y):
    """학습된 모델의 성능을 평가합니다."""
    d = xgb.DMatrix(X)
    y_prob = bst.predict(d)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0)
    }

# ---------------------------
# 3) 메인 실행 함수
# ---------------------------
def main():
    # 1. 데이터 로드 및 전처리 (data_utils.py 함수 사용)
    # 스크립트의 위치를 기준으로 데이터 파일의 절대 경로를 생성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, DATA_PATH)
    df = data_utils.load_and_clean(full_data_path)
    if df is None: return
    
    df = data_utils.define_label(df)
    df = data_utils.augment_data(df)
    df = data_utils.feature_engineer(df)
    
    # 2. 학습 데이터 준비 (data_utils.py 함수 사용)
    X, y, num_imputer, scaler = data_utils.prepare_training_data(df)
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] 데이터 분할 완료: train {len(X_train)} / test {len(X_test)}")
    
    # 4. 모델 학습
    print("\n[INFO] 모델 학습 시작...")
    bst = train_xgb(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # # 5. 모델 및 전처리 객체 저장
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(base_dir, MODEL_OUT)
    # model_pack = {
    #     "model": bst,
    #     "imputer": num_imputer,
    #     "scaler": scaler,
    #     "features": list(X.columns)
    # }
    # joblib.dump(model_pack, model_path)
    # print(f"\n[INFO] 모델 저장 완료: {model_path}")
    
    # 6. 모델 평가
    test_metrics = evaluate_model(bst, X_test, y_test)
    print("\n--- 모델 평가 결과 (Test Set) ---")
    print({k: f"{v:.4f}" for k, v in test_metrics.items()})

if __name__ == "__main__":
    main()

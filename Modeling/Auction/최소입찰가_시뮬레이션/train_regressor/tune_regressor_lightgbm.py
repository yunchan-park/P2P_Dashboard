# train_regressor_lightgbm_grid.py
"""
GridSearchCV를 사용하여 LightGBM 회귀 모델 하이퍼파라미터 튜닝 및 모델 저장.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# data_utils.py를 찾기 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


# -----------------------------
# 데이터 준비 함수
# -----------------------------
def prepare_regression_data(df):
    """회귀 모델 학습을 위한 최종 데이터셋(X, y)을 준비합니다."""
    features = [
        "감정가", "최저가", "유찰횟수", "건물면적", "토지면적",
        "건축연수", "면적당감정가", "최저비율", "층", "매각연도", "매각월"
    ]

    categorical_cols = ['bjd_sido', 'bjd_sigungu']

    y = df["낙찰가"]

    # 숫자형 처리
    X_num = df[features].copy()
    num_imputer = SimpleImputer(strategy="median").fit(X_num)
    X_num = pd.DataFrame(num_imputer.transform(X_num), columns=features, index=df.index)

    # 범주형 처리
    X_cat = pd.DataFrame(index=df.index)
    for c in categorical_cols:
        X_cat[c] = df[c].fillna("NA").astype("category").cat.codes

    X = pd.concat([X_num, X_cat], axis=1)

    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, num_imputer, scaler


# -----------------------------
# 평가 함수
# -----------------------------
def evaluate_model(model, X, y):
    pred = model.predict(X)
    return {
        "R2 Score": r2_score(y, pred),
        "MAE": mean_absolute_error(y, pred),
        "MAPE": mean_absolute_percentage_error(y, pred) * 100
    }


# -----------------------------
# 메인 함수
# -----------------------------
def main():
    RANDOM_STATE = 42
    DATA_PATH = "../../Data_Madang/auction_preprocessed.csv"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, DATA_PATH)

    # 데이터 로드
    df = data_utils.load_and_clean(full_data_path)
    df = data_utils.feature_engineer(df)

    df_success = df[df["낙찰가"].notnull()].copy()
    print(f"[INFO] 회귀 모델 학습 데이터 크기: {len(df_success)}")

    X, y, imputer, scaler = prepare_regression_data(df_success)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"[INFO] 데이터 분할 완료: train {len(X_train)} / test {len(X_test)}")

    # -----------------------------
    # GridSearchCV
    # -----------------------------
    model = lgb.LGBMRegressor(random_state=RANDOM_STATE)

    param_grid = {
        "num_leaves": [31, 50, 70],
        "max_depth": [-1, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [300, 600, 1000],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    print("\n[INFO] GridSearchCV 하이퍼파라미터 검색 시작...")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_percentage_error",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\n[INFO] GridSearchCV 완료!")
    print("[BEST PARAMS]", grid.best_params_)
    print("[BEST SCORE] (MAPE)", -grid.best_score_)

    best_model = grid.best_estimator_

    # -----------------------------
    # 테스트 평가
    # -----------------------------
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print("\n--- 회귀 모델 평가 결과 (Test Set) ---")
    for k, v in test_metrics.items():
        if k == "MAPE":
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")

    # -----------------------------
    # 최종 모델 저장
    # -----------------------------
    save_dir = os.path.join(base_dir, "../trained_model")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "auction_regressor_lgbm_tuned.joblib")

    model_pack = {
        "model": best_model,
        "imputer": imputer,
        "scaler": scaler,
        "features": list(X_train.columns)
    }

    joblib.dump(model_pack, model_path)
    print(f"\n[INFO] 최종 모델 저장 완료: {model_path}")


if __name__ == "__main__":
    main()

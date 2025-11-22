# train_regressor_randomforest.py
"""
낙찰된 경매 데이터를 사용하여 '낙찰가'를 예측하는 RandomForest 회귀 모델을 학습합니다.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# data_utils.py 경로 설정을 위해 상위 폴더를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils

# --- 경로 및 상수 정의 ---
DATA_PATH = "../../Data_Madang/auction_preprocessed.csv"
RANDOM_STATE = 42
# MODEL_OUT = "../trained_model/auction_regressor.joblib"

# --- 데이터 준비 함수 ---
def prepare_regression_data(df):
    """회귀 모델 학습을 위한 최종 데이터셋(X, y)을 준비합니다."""
    features = [
        "감정가", "최저가", "유찰횟수", "건물면적", "토지면적", 
        "건축연수", "면적당감정가", "최저비율", "층", "매각연도", "매각월"
    ]
    categorical_cols = ['bjd_sido', 'bjd_sigungu']
    y = df["낙찰가"]
    X_num = df[features].copy()
    num_imputer = SimpleImputer(strategy="median").fit(X_num)
    X_num = pd.DataFrame(num_imputer.transform(X_num), columns=features, index=df.index)
    X_cat = pd.DataFrame(index=df.index)
    for c in categorical_cols:
        X_cat[c] = df[c].fillna("NA").astype('category').cat.codes
    X = pd.concat([X_num, X_cat], axis=1)
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X_scaled, y, num_imputer, scaler

# --- 모델 학습 및 평가 함수 ---
def train_rf_regressor(X_train, y_train):
    """RandomForest 회귀 모델을 학습시킵니다."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=4,
        min_samples_split=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_regressor(model, X, y):
    """학습된 회귀 모델의 성능을 평가합니다."""
    y_pred = model.predict(X)
    return {
        "R2 Score": r2_score(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred) * 100
    }

# --- 메인 실행 함수 ---
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, DATA_PATH)
    df = data_utils.load_and_clean(full_data_path)
    if df is None: return
    
    df = data_utils.feature_engineer(df)
    
    df_success = df[df['낙찰가'].notnull()].copy()
    print(f"[INFO] 회귀 모델 학습을 위한 데이터 크기: {len(df_success)}")
    
    X, y, imputer, scaler = prepare_regression_data(df_success)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[INFO] 데이터 분할 완료: train {len(X_train)} / test {len(X_test)}")
    
    print("\n[INFO] 회귀 모델 학습 시작...")
    model = train_rf_regressor(X_train, y_train)
    
    # model_path = os.path.join(base_dir, MODEL_OUT)
    # model_pack = {
    #     "model": model,
    #     "imputer": imputer,
    #     "scaler": scaler,
    #     "features": list(X.columns)
    # }
    # joblib.dump(model_pack, model_path)
    # print(f"\n[INFO] 모델 저장 완료: {model_path}")
    
    test_metrics = evaluate_regressor(model, X_test, y_test)
    print("\n--- 회귀 모델 평가 결과 (Test Set) ---")
    for k, v in test_metrics.items():
        if k == "MAPE":
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

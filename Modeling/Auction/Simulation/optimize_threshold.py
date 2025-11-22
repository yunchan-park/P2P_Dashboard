"""
최적의 임계값(THRESHOLD)을 찾기 위한 스크립트.
0.1부터 0.9까지의 임계값에 대해 시뮬레이션을 반복 실행하고,
각 임계값별 MAPE를 계산하여 가장 좋은 성능을 내는 값을 찾습니다.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# data_utils.py 불러오기
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_utils

# ---------------------------
# 1) 설정
# ---------------------------
CLASSIFIER_MODEL_NAME = "trained_model/auction_classifier_xgb_tuned.joblib"
REGRESSOR_MODEL_NAME = "trained_model/auction_regressor_xgb_tuned.joblib"
DATA_PATH = "../Data_Madang/auction_preprocessed.csv"
RANDOM_STATE = 42

# ---------------------------
# 2) 시뮬레이터 함수 (기존 simulate.py와 동일)
# ---------------------------
def run_classification_simulation(item_row, model_pack, max_rounds=8):
    results = []
    model = model_pack["model"]
    imputer = model_pack["imputer"]
    scaler = model_pack["scaler"]
    feature_cols = model_pack["features"]
    num_cols = imputer.feature_names_in_
    cat_cols_in_model = [c for c in feature_cols if c not in num_cols]

    for k in range(max_rounds + 1):
        sim_item = item_row.copy()
        sim_item["유찰횟수"] = k
        sim_item["최저가"] = data_utils.calc_min_price_by_round(sim_item["감정가"], k, sim_item["소재지"])
        sim_df = data_utils.feature_engineer(pd.DataFrame([sim_item]))
        X_num = sim_df[num_cols]
        X_cat = pd.DataFrame(index=sim_df.index)
        for c in cat_cols_in_model:
            if c in sim_df.columns:
                X_cat[c] = sim_df[c].fillna("NA").astype("category").cat.codes
            else:
                X_cat[c] = 0
        X_num_imp = pd.DataFrame(imputer.transform(X_num), columns=num_cols)
        X_full = pd.concat([X_num_imp, X_cat], axis=1)[feature_cols]
        X_scaled = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)

        if X_scaled.isnull().sum().sum() > 0:
            X_scaled.fillna(0, inplace=True)

        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X_scaled)
            prob = model.predict(dmatrix)[0]
        else:
            prob = model.predict_proba(X_scaled)[:, 1][0]

        results.append({
            "round": k, "prob": float(prob), "data_for_prediction": sim_df.iloc[0]
        })
    return results

def predict_hammer_price(item_data_df, model_pack):
    model = model_pack["model"]
    imputer = model_pack["imputer"]
    scaler = model_pack["scaler"]
    feature_cols = model_pack["features"]
    
    # NaN 방어 코드 추가
    item_data_df.fillna(0, inplace=True)

    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(item_data_df[feature_cols])
        predicted_price = model.predict(dmatrix)[0]
    else:
        # 이 부분은 Scikit-learn 래퍼를 사용할 경우를 대비해 남겨둡니다.
        # 현재는 imputer, scaler를 직접 사용하지 않으므로,
        # 전처리된 데이터프레임이 들어온다는 가정 하에 predict를 수행합니다.
        predicted_price = model.predict(item_data_df[feature_cols])[0]
    return predicted_price

# ---------------------------
# 3) 메인 실행
# ---------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- 모델 로드 ---
    try:
        classifier_pack = joblib.load(os.path.join(base_dir, CLASSIFIER_MODEL_NAME))
        regressor_pack = joblib.load(os.path.join(base_dir, REGRESSOR_MODEL_NAME))
        print("[INFO] 모델 로드 완료.")
    except FileNotFoundError as e:
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {e.filename}")
        return
        
    # --- 데이터 준비 (한 번만 실행) ---
    print("[INFO] 데이터 준비 및 분할 중...")
    full_path = os.path.join(base_dir, DATA_PATH)
    df = data_utils.load_and_clean(full_path)
    if df is None: return
    df_with_label = data_utils.define_label(df)
    try:
        _, df_test_orig = train_test_split(
            df_with_label, test_size=0.2, random_state=RANDOM_STATE, stratify=df_with_label['label']
        )
    except ValueError:
        _, df_test_orig = train_test_split(
            df_with_label, test_size=0.2, random_state=RANDOM_STATE
        )
    df_test = df_test_orig[df_test_orig['낙찰가'].notna()].copy()
    if len(df_test) > 100:
        df_test = df_test.sample(n=100, random_state=RANDOM_STATE)
    print(f"[INFO] 최종 시뮬레이션 대상 테스트 케이스 수: {len(df_test)}")

    results_summary = []
    
    # --- 임계값(Threshold)을 변경하며 루프 실행 ---
    threshold_range = np.arange(0.1, 0.95, 0.05)
    print(f"\n[INFO] {threshold_range} 범위의 임계값에 대해 최적화 시작...")

    for threshold in threshold_range:
        actual_prices = []
        predicted_prices = []

        for _, item in df_test.iterrows():
            simulation_history = run_classification_simulation(item, classifier_pack)
            optimal_round_info = next((r for r in simulation_history if r["prob"] >= threshold), None)

            if optimal_round_info:
                # predict_hammer_price를 위한 데이터 준비
                item_for_reg_raw = optimal_round_info["data_for_prediction"]
                
                # 회귀 모델의 전처리를 여기서 직접 수행
                reg_imputer = regressor_pack["imputer"]
                reg_scaler = regressor_pack["scaler"]
                reg_features = regressor_pack["features"]
                
                # 원본 데이터프레임 생성
                item_df = pd.DataFrame([item_for_reg_raw])
                
                # 숫자형/범주형 분리
                X_num = item_df[reg_imputer.feature_names_in_]
                X_cat = pd.DataFrame(index=item_df.index)
                for c in [col for col in reg_features if col not in reg_imputer.feature_names_in_]:
                    if c in item_df.columns:
                        X_cat[c] = item_df[c].fillna("NA").astype("category").cat.codes
                    else:
                        X_cat[c] = 0

                # 전처리 수행
                X_num_imp = pd.DataFrame(reg_imputer.transform(X_num), columns=reg_imputer.feature_names_in_)
                X_full = pd.concat([X_num_imp, X_cat], axis=1)[reg_features]
                X_scaled = pd.DataFrame(reg_scaler.transform(X_full), columns=reg_features)
                
                # 예측
                predicted_price = predict_hammer_price(X_scaled, regressor_pack)

                actual_prices.append(item['낙찰가'])
                predicted_prices.append(predicted_price)
        
        mape = np.nan
        if actual_prices:
            actual_prices_arr = np.array(actual_prices)
            predicted_prices_arr = np.array(predicted_prices)
            mape = np.mean(np.abs((actual_prices_arr - predicted_prices_arr) / actual_prices_arr)) * 100
        
        print(f"  - Threshold: {threshold:.2f} | MAPE: {mape:.2f}% | 성공: {len(actual_prices)}/{len(df_test)}")
        results_summary.append({
            "Threshold": threshold,
            "MAPE": mape,
            "Success_Count": len(actual_prices)
        })

    # --- 최종 결과 출력 ---
    print("\n\n" + "#"*30 + " 최종 임계값 성능 비교 " + "#"*30)
    if results_summary:
        results_df = pd.DataFrame(results_summary).dropna()
        if not results_df.empty:
            results_df = results_df.sort_values(by="MAPE", ascending=True).reset_index(drop=True)
            print(results_df)

            best_result = results_df.iloc[0]
            print("\n--- 가장 우수한 임계값 ---")
            print(f"  임계값 (Threshold): {best_result['Threshold']:.2f}")
            print(f"  MAPE:               {best_result['MAPE']:.2f}%")
            print(f"  예측 성공률:        {best_result['Success_Count']}/{len(df_test)}")
        else:
            print("유효한 MAPE 결과를 얻지 못했습니다.")
    else:
        print("모든 임계값에 대해 시뮬레이션에 실패했습니다.")
    print("#"*82 + "\n")

if __name__ == "__main__":
    main()

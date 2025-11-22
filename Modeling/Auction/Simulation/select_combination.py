"""
분류기 3종과 회귀 모델 3종의 모든 조합(3x3=9가지)에 대해 시뮬레이션을 실행하고,
각 조합의 성능(MAPE)을 비교하여 최적의 조합을 찾습니다.
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
# 테스트할 모델 파일 목록
CLASSIFIER_MODEL_PATHS = {
    "LGBM_Classifier": "trained_model/auction_classifier_lgbm_tuned.joblib",
    "RF_Classifier": "trained_model/auction_classifier_rf_tuned.joblib",
    "XGB_Classifier": "trained_model/auction_classifier_xgb_tuned.joblib",
}
REGRESSOR_MODEL_PATHS = {
    "LGBM_Regressor": "trained_model/auction_regressor_lgbm_tuned.joblib",
    "RF_Regressor": "trained_model/auction_regressor_rf_tuned.joblib",
    "XGB_Regressor": "trained_model/auction_regressor_xgb_tuned.joblib",
}
DATA_PATH = "../Data_Madang/auction_preprocessed.csv"
THRESHOLD = 0.40
RANDOM_STATE = 42

# --------------------------- 
# 2) 시뮬레이터 함수
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
    num_cols = imputer.feature_names_in_
    cat_cols_in_model = [c for c in feature_cols if c not in num_cols]
    
    X_num = item_data_df[num_cols]
    X_cat = pd.DataFrame(index=item_data_df.index)
    for c in cat_cols_in_model:
        if c in item_data_df.columns:
            X_cat[c] = item_data_df[c].fillna("NA").astype("category").cat.codes
        else:
            X_cat[c] = 0
    X_num_imp = pd.DataFrame(imputer.transform(X_num), columns=num_cols)
    X_full = pd.concat([X_num_imp, X_cat], axis=1)[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)

    if X_scaled.isnull().sum().sum() > 0:
        X_scaled.fillna(0, inplace=True)

    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(X_scaled)
        predicted_price = model.predict(dmatrix)[0]
    else:
        predicted_price = model.predict(X_scaled)[0]
    return predicted_price

# --------------------------- 
# 3) 메인 실행
# --------------------------- 
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- 데이터 로드 및 준비 (한 번만 실행) ---
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

    # --- 9개 조합에 대한 루프 실행 ---
    for clf_name, clf_path in CLASSIFIER_MODEL_PATHS.items():
        for reg_name, reg_path in REGRESSOR_MODEL_PATHS.items():
            
            print("\n" + "="*70)
            print(f"Testing Combination: [Classifier: {clf_name}] + [Regressor: {reg_name}]")
            print("="*70)

            try:
                classifier_pack = joblib.load(os.path.join(base_dir, clf_path))
                regressor_pack = joblib.load(os.path.join(base_dir, reg_path))
                print("[INFO] 모델 로드 완료.")
            except FileNotFoundError as e:
                print(f"[오류] 모델 파일을 찾을 수 없습니다: {e.filename}")
                continue

            actual_prices = []
            predicted_prices = []

            for _, item in df_test.iterrows():
                simulation_history = run_classification_simulation(item, classifier_pack)
                optimal_round_info = next((r for r in simulation_history if r["prob"] >= THRESHOLD), None)

                if optimal_round_info:
                    item_for_reg = pd.DataFrame([optimal_round_info["data_for_prediction"]])
                    predicted_price = predict_hammer_price(item_for_reg, regressor_pack)
                    actual_prices.append(item['낙찰가'])
                    predicted_prices.append(predicted_price)
            
            if actual_prices:
                actual_prices_arr = np.array(actual_prices)
                predicted_prices_arr = np.array(predicted_prices)
                mape = np.mean(np.abs((actual_prices_arr - predicted_prices_arr) / actual_prices_arr)) * 100
                print(f"MAPE: {mape:.2f}% (성공: {len(actual_prices)}/{len(df_test)})")
                results_summary.append({
                    "Classifier": clf_name,
                    "Regressor": reg_name,
                    "MAPE": mape
                })
            else:
                print("가격 예측에 모두 실패했습니다.")

    # --- 최종 결과 출력 ---
    print("\n\n" + "#"*30 + " 최종 조합 성능 비교 " + "#"*30)
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        # MAPE 기준으로 정렬
        results_df = results_df.sort_values(by="MAPE", ascending=True).reset_index(drop=True)
        print(results_df)

        best_combo = results_df.iloc[0]
        print("\n--- 가장 우수한 조합 ---")
        print(f"  분류기 (Classifier): {best_combo['Classifier']}")
        print(f"  회귀 (Regressor):    {best_combo['Regressor']}")
        print(f"  MAPE:                {best_combo['MAPE']:.2f}%")
    else:
        print("모든 조합의 시뮬레이션에 실패했습니다.")
    print("#"*82 + "\n")

if __name__ == "__main__":
    main()

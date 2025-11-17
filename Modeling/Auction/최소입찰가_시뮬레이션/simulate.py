# simulate.py
"""
학습된 모델을 불러와, 테스트 세트의 특정 케이스들에 대한 실제 결과와
시뮬레이션 예측 결과를 비교하여 보여줍니다.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# data_utils.py 에서 필요한 함수들을 가져옵니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_utils

# ---------------------------
# 1) 설정
# ---------------------------
CLASSIFIER_MODEL_NAME = "trained_model/auction_classifier_lgbm_tuned.joblib"
REGRESSOR_MODEL_NAME = "trained_model/auction_regressor_lgbm_tuned.joblib"
DATA_PATH = "../Data_Madang/auction_preprocessed.csv"
THRESHOLD = 0.40
RANDOM_STATE = 42

# ---------------------------
# 2) 시뮬레이터 함수 (기존과 동일)
# ---------------------------
def run_classification_simulation(item_row, model_pack, max_rounds=8):
    results = []
    model, imputer, scaler, feature_cols = model_pack['model'], model_pack['imputer'], model_pack['scaler'], model_pack['features']
    num_cols = imputer.feature_names_in_
    cat_cols_in_model = [c for c in feature_cols if c not in num_cols]

    for k in range(max_rounds + 1):
        sim_item = item_row.copy()
        sim_item['유찰횟수'] = k
        sim_item['최저가'] = data_utils.calc_min_price_by_round(sim_item.get("감정가"), k, sim_item.get("소재지"))
        sim_df = data_utils.feature_engineer(pd.DataFrame([sim_item]))
        
        X_num = sim_df[num_cols]
        X_cat = pd.DataFrame(index=sim_df.index)
        for c in cat_cols_in_model:
            X_cat[c] = sim_df[c].fillna("NA").astype('category').cat.codes if c in sim_df.columns else 0
        
        X_num_imp = pd.DataFrame(imputer.transform(X_num), columns=num_cols)
        X_full = pd.concat([X_num_imp, X_cat], axis=1)[feature_cols]
        X_scaled = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)
        
        prob = model.predict(xgb.DMatrix(X_scaled))[0]
        results.append({"round": k, "min_price": sim_item['최저가'], "prob": float(prob), "data_for_prediction": sim_df.iloc[0]})
    return results

def predict_hammer_price(item_data_df, model_pack):
    model, imputer, scaler, feature_cols = model_pack['model'], model_pack['imputer'], model_pack['scaler'], model_pack['features']
    num_cols = imputer.feature_names_in_
    cat_cols_in_model = [c for c in feature_cols if c not in num_cols]

    X_num = item_data_df[num_cols]
    X_cat = pd.DataFrame(index=item_data_df.index)
    for c in cat_cols_in_model:
        X_cat[c] = item_data_df[c].fillna("NA").astype('category').cat.codes if c in item_data_df.columns else 0

    X_num_imp = pd.DataFrame(imputer.transform(X_num), columns=num_cols)
    X_full = pd.concat([X_num_imp, X_cat], axis=1)[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)
    
    predicted_price = model.predict(xgb.DMatrix(X_scaled))[0]
    return predicted_price

# ---------------------------
# 3) 메인 실행 함수
# ---------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 모델 로드
    try:
        classifier_pack = joblib.load(os.path.join(base_dir, CLASSIFIER_MODEL_NAME))
        regressor_pack = joblib.load(os.path.join(base_dir, REGRESSOR_MODEL_NAME))
        print("[INFO] 모든 모델 로드 완료.")
    except FileNotFoundError as e:
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {e.filename}")
        return
    
    # 2. 학습과 동일한 방식으로 데이터 준비 및 분할
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, DATA_PATH)
    df = data_utils.load_and_clean(full_data_path)
    # full_data_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "Data_Madang/auction_preprocessed.csv")
    # df = data_utils.load_and_clean(full_data_path)
    if df is None: return

    # '절차적 종료' 케이스를 제외하고, 실제 낙찰된 데이터만 사용
    df_filtered = df[df['낙찰가'].notnull()].copy()

    df_with_label = data_utils.define_label(df_filtered)
    df_aug = data_utils.augment_data(df_with_label)
    df_featured = data_utils.feature_engineer(df_aug)
    
    X, y, _, _ = data_utils.prepare_training_data(df_featured)
    
    # 원본 데이터프레임의 인덱스를 사용하여 테스트셋 분리
    indices = df_featured.index
    _, X_test_indices, _, _ = train_test_split(indices, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    # 필터링된 원본 데이터에서 테스트 케이스 선택
    df_test = df_with_label.loc[df_with_label.index.isin(X_test_indices)]

    # 3. 케이스별 데이터 선정 ('최종 유찰' 케이스 제외)
    cases = {
        "0회 유찰 후 즉시 낙찰": df_test[df_test['유찰횟수'] == 0],
        "1회 유찰 후 낙찰": df_test[df_test['유찰횟수'] == 1],
        "2회 유찰 후 낙찰": df_test[df_test['유찰횟수'] == 2],
        "3회 이상 유찰 후 낙찰": df_test[df_test['유찰횟수'] >= 3],
    }
    
    selected_examples = {}
    for name, data in cases.items():
        if not data.empty:
            selected_examples[name] = data.iloc[0]

    # 4. 각 케이스에 대한 시뮬레이션 실행 및 결과 비교
    for name, item in selected_examples.items():
        print("\n" + "="*60)
        print(f"CASE: {name}")
        print("="*60)
        
        # 실제 결과 출력
        print("\n--- [실제 결과] ---")
        print(f"  - 소재지: {item['소재지']}")
        print(f"  - 감정가: {item['감정가']:,.0f} 원")
        print(f"  - 실제 유찰횟수: {int(item['유찰횟수'])}회")
        print(f"  - 실제 최저가: {item['최저가']:,.0f} 원")
        print(f"  - 최종 낙찰가: {item['낙찰가']:,.0f} 원 (낙찰가율: {item['낙찰가']/item['감정가']:.2%})")

        # 시뮬레이션 실행
        print("\n--- [시뮬레이션 예측] ---")
        item_for_sim = item.copy()
        item_for_sim['유찰횟수'] = 0
        
        simulation_history = run_classification_simulation(item_for_sim, classifier_pack)
        
        for r in simulation_history:
            print(f"  - Round {r['round']}: 최저가 {r['min_price']:,.0f} 원, 낙찰 확률: {r['prob']:.2%}")
        
        # 최적 회차 찾기 (단순 기준점 방식)
        optimal_round_info = next((r for r in simulation_history if r['prob'] >= THRESHOLD), None)
        
        # 최종 결과 출력
        if optimal_round_info:
            print(f"\n  ==> 1차 예측: Round {optimal_round_info['round']} 에서 낙찰 확률({optimal_round_info['prob']:.2%})이 기준점({THRESHOLD:.0%})을 넘을 것으로 예상됩니다.")
            item_for_regression = pd.DataFrame([optimal_round_info['data_for_prediction']])
            predicted_price = predict_hammer_price(item_for_regression, regressor_pack)
            print(f"  ==> 최종 예측: 최적 입찰 시점(Round {optimal_round_info['round']})을 고려한 예상 낙찰가는 **{predicted_price:,.0f} 원** 입니다.")
        else:
            print(f"\n  ==> 최종 예측: 시뮬레이션 내에서는 낙찰 확률이 기준점({THRESHOLD:.0%})을 넘지 못할 것으로 예상됩니다.")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    main()

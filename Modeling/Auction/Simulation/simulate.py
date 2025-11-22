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
CLASSIFIER_MODEL_NAME = "trained_model/auction_classifier_xgb.joblib"
REGRESSOR_MODEL_NAME = "trained_model/auction_regressor_xgb.joblib"
DATA_PATH = "../Data_Madang/auction_preprocessed.csv"
THRESHOLD = 0.40
RANDOM_STATE = 42

# ---------------------------
# 2) 시뮬레이터 함수
# ---------------------------
def run_classification_simulation(item_row, model_pack, max_rounds=8):
    results = []

    # 저장된 preprocessing 요소들
    model = model_pack["model"]
    imputer = model_pack["imputer"]
    scaler = model_pack["scaler"]
    feature_cols = model_pack["features"]

    num_cols = imputer.feature_names_in_
    cat_cols_in_model = [c for c in feature_cols if c not in num_cols]

    for k in range(max_rounds + 1):
        sim_item = item_row.copy()

        # 유찰 횟수 설정
        sim_item["유찰횟수"] = k

        # 최저가 업데이트
        sim_item["최저가"] = data_utils.calc_min_price_by_round(
            sim_item["감정가"], k, sim_item["소재지"]
        )

        # feature engineering 통과
        sim_df = data_utils.feature_engineer(pd.DataFrame([sim_item]))

        # 숫자/범주형 분리
        X_num = sim_df[num_cols]

        # 범주형 처리 (코드화)
        X_cat = pd.DataFrame(index=sim_df.index)
        for c in cat_cols_in_model:
            if c in sim_df.columns:
                X_cat[c] = sim_df[c].fillna("NA").astype("category").cat.codes
            else:
                X_cat[c] = 0

        # 전처리(결측치 → impute)
        X_num_imp = pd.DataFrame(imputer.transform(X_num), columns=num_cols)

        # 전체 feature 순서 맞추기
        X_full = pd.concat([X_num_imp, X_cat], axis=1)[feature_cols]

        # 스케일링
        X_scaled = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)

        # [수정] NaN 값을 허용하지 않는 모델을 위해 최종 단계에서 한번 더 처리
        if X_scaled.isnull().sum().sum() > 0:
            X_scaled.fillna(0, inplace=True)

        # 모델별 예측 (확률)
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X_scaled)
            prob = model.predict(dmatrix)[0]
        else:
            prob = model.predict_proba(X_scaled)[:,1][0]

        results.append({
            "round": k,
            "min_price": sim_item["최저가"],
            "prob": float(prob),
            "data_for_prediction": sim_df.iloc[0]
        })

    return results


def predict_hammer_price(item_data_df, model_pack):
    model = model_pack["model"]
    imputer = model_pack["imputer"]
    scaler = model_pack["scaler"]
    feature_cols = model_pack["features"]

    num_cols = imputer.feature_names_in_
    cat_cols_in_model = [c for c in feature_cols if c not in num_cols]

    # 숫자 컬럼
    X_num = item_data_df[num_cols]

    # 범주형
    X_cat = pd.DataFrame(index=item_data_df.index)
    for c in cat_cols_in_model:
        if c in item_data_df.columns:
            X_cat[c] = item_data_df[c].fillna("NA").astype("category").cat.codes
        else:
            X_cat[c] = 0

    # 결측치 처리
    X_num_imp = pd.DataFrame(imputer.transform(X_num), columns=num_cols)

    # 전체 feature
    X_full = pd.concat([X_num_imp, X_cat], axis=1)[feature_cols]

    # 스케일링
    X_scaled = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)
    
    # [수정] NaN 값을 허용하지 않는 모델을 위해 최종 단계에서 한번 더 처리
    if X_scaled.isnull().sum().sum() > 0:
        X_scaled.fillna(0, inplace=True)

    # 모델별 예측
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

    print("[INFO] 모델 로딩 중...")
    try:
        classifier_pack = joblib.load(os.path.join(base_dir, CLASSIFIER_MODEL_NAME))
        regressor_pack = joblib.load(os.path.join(base_dir, REGRESSOR_MODEL_NAME))
        print("[INFO] 모든 모델 로드 완료.")
    except FileNotFoundError as e:
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {e.filename}")
        return

    print("[INFO] 데이터 준비 및 분할 중...")
    full_path = os.path.join(base_dir, DATA_PATH)
    df = data_utils.load_and_clean(full_path)
    if df is None: return

    df_with_label = data_utils.define_label(df)

    
    # 데이터 증강 전에 원본 데이터에서 훈련/테스트셋 분리
    try:
        df_train_orig, df_test_orig = train_test_split(
            df_with_label, test_size=0.2, random_state=RANDOM_STATE, stratify=df_with_label['label']
        )
    except ValueError: # stratify가 불가능할 경우 (클래스 1개)
         df_train_orig, df_test_orig = train_test_split(
            df_with_label, test_size=0.2, random_state=RANDOM_STATE
        )

    # 시뮬레이션 대상인 테스트셋은 실제 낙찰가가 있는 데이터만 사용
    df_test = df_test_orig[df_test_orig['낙찰가'].notna()].copy()
    
    # 100개만 랜덤 샘플링하여 시간 단축
    # if len(df_test) > 100:
    #     print(f"[INFO] 테스트셋이 100개를 초과하여 {len(df_test)}개 중 100개만 랜덤 샘플링합니다.")
    #     df_test = df_test.sample(n=100, random_state=RANDOM_STATE)

    print(f"[INFO] 최종 시뮬레이션 대상 테스트 케이스 수: {len(df_test)}")
    
    # 상세히 출력할 4가지 대표 케이스 선정 (테스트셋 내에서 선정)
    representative_cases = {}
    if not df_test[df_test["유찰횟수"] == 0].empty:
        representative_cases["0회 유찰 후 즉시 낙찰"] = df_test[df_test["유찰횟수"] == 0].iloc[0].name
    if not df_test[df_test["유찰횟수"] == 1].empty:
        representative_cases["1회 유찰 후 낙찰"] = df_test[df_test["유찰횟수"] == 1].iloc[0].name
    if not df_test[df_test["유찰횟수"] == 2].empty:
        representative_cases["2회 유찰 후 낙찰"] = df_test[df_test["유찰횟수"] == 2].iloc[0].name
    if not df_test[df_test["유찰횟수"] >= 3].empty:
        representative_cases["3회 이상 유찰 후 낙찰"] = df_test[df_test["유찰횟수"] >= 3].iloc[0].name

    # MAPE 계산을 위한 리스트 초기화
    actual_prices = []
    predicted_prices = []
    
    # --- 전체 테스트셋에 대한 시뮬레이션 실행 (단일 루프) ---
    print("\n[INFO] 전체 테스트셋에 대한 시뮬레이션 시작...")
    
    for i, (index, item) in enumerate(df_test.iterrows()):
        
        # 상세 출력 여부 결정
        should_print_details = False
        case_name_for_print = ""
        for name, rep_idx in representative_cases.items():
            if index == rep_idx:
                should_print_details = True
                case_name_for_print = name
                break

        if should_print_details:
            print("\n" + "=" * 60)
            print(f"CASE: {case_name_for_print}")
            print("=" * 60)
            print("\n--- [실제 결과] ---")
            print(f"  - 소재지: {item['소재지']}")
            print(f"  - 감정가: {item['감정가']:,.0f} 원")
            print(f"  - 실제 유찰횟수: {int(item['유찰횟수'])}회")
            print(f"  - 실제 최저가: {item['최저가']:,.0f} 원")
            actual_hammer_price = item['낙찰가']
            print(f"  - 실제 낙찰가: {actual_hammer_price:,.0f} 원 (낙찰가율: {actual_hammer_price / item['감정가']:.2%})")
            print("\n--- [XGBoost 시뮬레이션 예측] ---")

        item_for_sim = item.copy()
        item_for_sim["유찰횟수"] = 0
        simulation_history = run_classification_simulation(item_for_sim, classifier_pack)

        if should_print_details:
            for r in simulation_history:
                print(f"  - Round {r['round']}: 최저가 {r['min_price']:,.0f} 원 | 낙찰 확률: {r['prob']:.2%}")
        
        optimal_round_info = next((r for r in simulation_history if r["prob"] >= THRESHOLD), None)

        if optimal_round_info:
            item_for_reg = pd.DataFrame([optimal_round_info["data_for_prediction"]])
            predicted_price = predict_hammer_price(item_for_reg, regressor_pack)
            
            actual_prices.append(item['낙찰가'])
            predicted_prices.append(predicted_price)
            
            if should_print_details:
                print("\n  ==> 최적 회차:", optimal_round_info["round"])
                print(f"      (확률 {optimal_round_info['prob']:.2%} ≥ 기준 {THRESHOLD:.0%})")
                print(f"  ==> 예상 낙찰가: {predicted_price:,.0f} 원")
        
        elif should_print_details: # 예측은 못했지만, 상세 출력 대상인 경우
            print(f"\n  ==> 어떤 회차에서도 낙찰 확률이 기준({THRESHOLD:.0%})을 넘지 못함")

        if should_print_details:
             print("=" * 60 + "\n")
    
    print("[INFO] 시뮬레이션 완료.")

    # 최종 MAPE 계산 및 출력
    if actual_prices:
        actual_prices = np.array(actual_prices)
        predicted_prices = np.array(predicted_prices)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        print("\n" + "#" * 20 + " 최종 MAPE 결과 " + "#" * 20)
        print(f"--- 최종 조합 (Classifier: XGB, Regressor: XGB) ---")
        print(f"전체 시뮬레이션 대상 케이스 수: {len(df_test)}")
        print(f"가격 예측이 성공한 케이스 수 (MAPE 계산 대상): {len(actual_prices)}")
        print(f"MAPE: {mape:.2f}%")
        print("#" * 70 + "\n")
    else:
        print("모든 시뮬레이션 대상 테스트 케이스에 대해 가격 예측을 하지 못했습니다.")

if __name__ == "__main__":
    main()

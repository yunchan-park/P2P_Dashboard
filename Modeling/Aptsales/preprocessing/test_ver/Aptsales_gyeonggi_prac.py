import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from category_encoders import TargetEncoder, OneHotEncoder

# --- 0. 최종 모델링 스크립트 ---
# 이 스크립트는 논의된 모든 개선 사항을 반영합니다:
# 1. 상대 경로 사용
# 2. 면적 구간 피처 생성 및 세분화된 그룹핑
# 3. 시계열 분할 검증
# 4. MAPE 및 지역구별 상세 평가 지표 추가

def create_time_series_features(df):
    """단지명과 면적구간으로 그룹화하여 시계열 피처를 생성합니다."""
    print("\n[INFO] Creating area groups and time-series features...")
    # 면적구간 피처 생성
    bins = [0, 60, 85, 135, float('inf')]
    labels = ['소형', '중형', '대형', '초대형']
    df['면적구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    if '계약년월' not in df.columns:
        print("[WARNING] '계약년월' column not found. Skipping time-series features.")
        return df
    
    df['계약년월'] = pd.to_datetime(df['계약년월'], format='mixed')
    df['계약년'] = df['계약년월'].dt.year
    df['계약월'] = df['계약년월'].dt.month
    
    df = df.sort_values(by=['단지명', '면적구간', '계약년월']).reset_index(drop=True)
    
    # EWMA 피처 (그룹화 기준 변경)
    df['price_ewma_3'] = df.groupby(['단지명', '면적구간'], observed=False)['거래금액(만원)'].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    
    # 결측치 처리 (그룹화 기준 변경)
    df['price_ewma_3'] = df.groupby(['단지명', '면적구간'], observed=False)['price_ewma_3'].transform(lambda x: x.bfill())
    df['price_ewma_3'] = df['price_ewma_3'].fillna(0)
    
    print("[INFO] Feature creation complete.")
    return df

def train_and_evaluate(df, version_name):
    """시계열 분할, 상세 평가 지표를 포함한 모델 학습 및 평가 함수"""
    print(f"\n--- [Version: {version_name}] Starting Model Training & Evaluation (Data size: {len(df)}) ---")

    # --- 시계열 분할 ---
    print("\n[INFO] Performing time-series split (80% train, 20% test)...")
    df_sorted = df.sort_values(by='계약년월').reset_index(drop=True)
    split_point = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_point]
    test_df = df_sorted.iloc[split_point:]

    print(f"Train period: {train_df['계약년월'].min().strftime('%Y-%m')} to {train_df['계약년월'].max().strftime('%Y-%m')}")
    print(f"Test period:  {test_df['계약년월'].min().strftime('%Y-%m')} to {test_df['계약년월'].max().strftime('%Y-%m')}")

    y_train = train_df['거래금액(만원)']
    y_test = test_df['거래금액(만원)']
    
    cols_to_drop = ['NO', '거래금액(만원)', '계약년월']
    X_train = train_df.drop(cols_to_drop, axis=1)
    X_test = test_df.drop(cols_to_drop, axis=1)

    # --- 인코딩 ---
    print("[INFO] Applying encoders...")
    te = TargetEncoder(cols=['단지명', '면적구간'])
    X_train_encoded = te.fit_transform(X_train, y_train)
    X_test_encoded = te.transform(X_test)

    X_train_encoded = pd.get_dummies(X_train_encoded, columns=['시군구'], prefix='시군구')
    X_test_encoded = pd.get_dummies(X_test_encoded, columns=['시군구'], prefix='시군구')

    X_train_final = X_train_encoded
    X_test_final = X_test_encoded.reindex(columns=X_train_final.columns, fill_value=0)

    # --- 모델 학습 및 평가 ---
    print("[INFO] Training and evaluating models...")
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1)
    }
    
    overall_results = []
    best_model_info = {'model': None, 'name': '', 'mae': float('inf')}

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_final, y_train)
        predictions = model.predict(X_test_final)
        end_time = time.time()

        mae = mean_absolute_error(y_test, predictions)
        if mae < best_model_info['mae']:
            best_model_info['model'] = model
            best_model_info['name'] = name
            best_model_info['mae'] = mae

        overall_results.append({
            'Model': name,
            'MAE': mae,
            'MAPE (%)': mean_absolute_percentage_error(y_test, predictions) * 100,
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'R2': r2_score(y_test, predictions),
            'Time(s)': end_time - start_time
        })
    
    # --- 최종 결과 출력 ---
    print("\n--- Overall Model Performance ---")
    results_df = pd.DataFrame(overall_results).sort_values(by='MAE')
    print(results_df.to_string())

    # --- 지역구별 상세 평가 ---
    print(f"\n--- Detailed Performance by District (Gu) for Best Model: {best_model_info['name']} ---")
    eval_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': best_model_info['model'].predict(X_test_final),
        '시군구': test_df['시군구']
    })
    
    eval_df['구'] = eval_df['시군구'].str.split(' ').str[1]
    
    gu_results = []
    for gu, group in eval_df.groupby('구'):
        gu_mae = mean_absolute_error(group['y_true'], group['y_pred'])
        gu_mape = mean_absolute_percentage_error(group['y_true'], group['y_pred']) * 100
        gu_results.append({'District': gu, 'MAE': gu_mae, 'MAPE (%)': gu_mape, 'Count': len(group)})
        
    gu_results_df = pd.DataFrame(gu_results).sort_values(by='MAE', ascending=False)
    print(gu_results_df.to_string())

    # --- 지역구별 평가 결과 엑셀 저장 ---
    try:
        gu_results_df.to_excel('../../../../경기도_구별_평가_결과.xlsx', index=False)
        print("\n[INFO] '경기도_구별_평가_결과.xlsx' 파일이 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"\n[ERROR] 엑셀 파일 저장 중 오류가 발생했습니다: {e}")

    return best_model_info, te, X_train_final.columns

# --- 1. 데이터 로드 및 전처리 ---
try:
    # CSV 파일 사용으로 변경
    data_path = '../../data/경기_매매_합본.csv'
    df_original = pd.read_csv(data_path, encoding='utf-8')
except FileNotFoundError:
    print(f"[ERROR] Data file not found at: {data_path}. Please ensure you have saved the excel as '경기도_매매_데이터.csv' with UTF-8 encoding.")
    exit()

df_original['거래금액(만원)'] = df_original['거래금액(만원)'].astype(str).str.replace(',', '').astype(int)
df_original['가계대출_금리'] = df_original['가계대출_금리'].fillna(df_original['가계대출_금리'].mean())

# 시계열 피처 생성
df_with_features = create_time_series_features(df_original)

best_model_info, te, columns = train_and_evaluate(df_with_features, "Final Model")

# --- 3. 최적 모델 및 전처리기 저장 ---
best_model = best_model_info['model']
best_model_name = best_model_info['name']

print(f'\n--- Best model is "{best_model_name}". Saving model and encoders... ---')

joblib.dump(best_model, '../../predict_model/test_ver/rf_model_gyeonggi_prac.joblib')
joblib.dump(te, '../../predict_model/test_ver/target_encoder_gyeonggi_prac.joblib')
joblib.dump(columns, '../../predict_model/test_ver/model_columns_gyeonggi_prac.joblib')

print('\n--- Model and encoders have been saved successfully. ---')

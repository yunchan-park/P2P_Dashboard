import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder, OneHotEncoder

# --- 0. 최종 모델 훈련 스크립트 ---
# 이 스크립트는 전체 데이터를 사용하여 최종 모델을 훈련하고 저장합니다.

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

def train_final_model(df):
    """전체 데이터를 사용하여 최종 모델을 훈련합니다."""
    print(f"\n--- Starting Final Model Training (Data size: {len(df)}) ---")

    # --- 전체 데이터를 훈련에 사용 ---
    y_train = df['거래금액(만원)']
    cols_to_drop = ['NO', '거래금액(만원)', '계약년월']
    X_train = df.drop(cols_to_drop, axis=1)

    # --- 인코딩 ---
    print("[INFO] Applying encoders to the entire dataset...")
    te = TargetEncoder(cols=['단지명', '면적구간'])
    X_train_encoded = te.fit_transform(X_train, y_train)

    X_train_encoded = pd.get_dummies(X_train_encoded, columns=['시군구'], prefix='시군구')

    X_train_final = X_train_encoded

    # --- 최종 모델(RandomForest) 훈련 ---
    print("[INFO] Training the final RandomForest model on the entire dataset...")
    start_time = time.time()
    
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train_final, y_train)
    
    end_time = time.time()
    print(f"[INFO] Final model training complete. Time taken: {end_time - start_time:.2f} seconds.")

    return final_model, te, X_train_final.columns

# --- 1. 데이터 로드 및 전처리 ---
try:
    data_path = '../data/부산_매매_합본.csv'
    df_original = pd.read_csv(data_path, encoding='utf-8')
except FileNotFoundError:
    print(f"[ERROR] Data file not found at: {data_path}. Please ensure it exists.")
    exit()

df_original['거래금액(만원)'] = df_original['거래금액(만원)'].astype(str).str.replace(',', '').astype(int)
df_original['가계대출_금리'] = df_original['가계대출_금리'].fillna(df_original['가계대출_금리'].mean())

# 시계열 피처 생성
df_with_features = create_time_series_features(df_original)

final_model, te, columns = train_final_model(df_with_features)

# --- 3. 최종 모델 및 전처리기 저장 ---
print('\n--- Saving final model and encoders... ---')

joblib.dump(final_model, '../predict_model/rf_model_busan.joblib')
joblib.dump(te, '../predict_model/target_encoder_busan.joblib')
joblib.dump(columns, '../predict_model/model_columns_busan.joblib')

print('\n--- Final model and encoders have been saved successfully. ---')

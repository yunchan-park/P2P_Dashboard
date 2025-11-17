# data_utils.py
"""
데이터 처리 관련 유틸리티 함수 모음
- 데이터 로드, 정제, 증강, 피처 엔지니어링, 학습 데이터 준비
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- 유틸리티 함수 ---
R_AVG = 0.7295 # 데이터 기반 평균 r값

def parse_date(s):
    """날짜 형식의 문자열을 datetime 객체로 변환합니다."""
    try:
        return pd.to_datetime(s, format="%Y.%m.%d", errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def get_rule_based_r(location_str):
    """지역명 문자열을 기반으로 규칙에 따른 r값을 반환합니다."""
    if not isinstance(location_str, str):
        return R_AVG # 기본값
    
    if '서울' in location_str:
        return 0.8
    elif '인천' in location_str or '경기' in location_str:
        return 0.7
    else:
        return R_AVG

def calc_min_price_by_round(appraisal_price, round_k, location_str):
    """유찰 회차와 지역에 따른 최저입찰가를 계산합니다."""
    if pd.isna(appraisal_price):
        return np.nan
    r = get_rule_based_r(location_str)
    return appraisal_price * (r ** round_k)

def korean_currency_to_float(s):
    """'억', '만' 단위가 포함된 금액 문자열을 숫자로 변환합니다."""
    try:
        s = str(s).replace(',', '').replace('원', '').strip()
        if not s or s == 'nan': return np.nan
        total = 0
        if '억' in s:
            parts = s.split('억'); total += float(parts[0]) * 100000000; s = parts[1]
        if '만' in s:
            parts = s.split('만');
            if parts[0]: total += float(parts[0]) * 10000
        elif s:
            total += float(s)
        return total
    except (ValueError, IndexError):
        return np.nan

# --- 데이터 처리 파이프라인 함수 ---

def load_and_clean(path):
    """데이터를 로드하고 기본적인 숫자/날짜 변환을 수행합니다."""
    try:
        df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
        return None
    df.columns = [c.strip() for c in df.columns]
    
    currency_cols = ["감정가", "최저가", "낙찰가"]
    numeric_cols = ["건물면적", "토지면적", "건축년도", "유찰횟수", "층", "법정동코드"]
    
    for col in currency_cols:
        if col in df.columns: df[col] = df[col].apply(korean_currency_to_float)
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if "매각기일" in df.columns:
        df["매각기일"] = df["매각기일"].apply(parse_date)
        
    if '유찰횟수' in df.columns:
        df['유찰횟수'].fillna(0, inplace=True)
        
    return df

def define_label(df):
    """'낙찰가' 존재 여부에 따라 label을 0 또는 1로 정의합니다."""
    df["label"] = df.get("낙찰가").notnull().astype(int)
    return df

def augment_data(df):
    """성공한 경매 건에 대해 이전의 실패 기록을 가상으로 생성하여 데이터를 증강합니다."""
    print(f"[INFO] 원본 데이터 크기: {len(df)}")
    augmented_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get('유찰횟수')) or pd.isna(row.get('감정가')):
            continue
        
        if row['label'] == 1: # 낙찰된 건
            failure_rounds = int(row['유찰횟수'])
            for k in range(failure_rounds):
                new_row = row.copy()
                new_row['유찰횟수'] = k
                new_row['최저가'] = calc_min_price_by_round(row['감정가'], k, row['소재지'])
                new_row['label'] = 0
                augmented_rows.append(new_row)
            augmented_rows.append(row)
        else: # 원래부터 유찰/실패인 건
            augmented_rows.append(row)
            
    augmented_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
    print(f"[INFO] 데이터 증강 후 크기: {len(augmented_df)}")
    return augmented_df

def feature_engineer(df):
    """기존 변수를 조합하여 새로운 파생 변수를 생성합니다."""
    if "매각기일" in df.columns:
        df["매각연도"] = df["매각기일"].dt.year
        df["매각월"] = df["매각기일"].dt.month
        if "건축년도" in df.columns:
            df["건축연수"] = df["매각연도"] - df["건축년도"]
    
    if "감정가" in df.columns and "건물면적" in df.columns:
        df["면적당감정가"] = df["감정가"] / df["건물면적"]
    
    if "최저가" in df.columns and "감정가" in df.columns:
        df["최저비율"] = df["최저가"] / df["감정가"]
        
    if "법정동코드" in df.columns:
        df['bjd_sido'] = (df['법정동코드'] // 100000000).astype(str)
        df['bjd_sigungu'] = (df['법정동코드'] // 100000).astype(str)
        
    return df

def prepare_training_data(df):
    """모델 학습을 위한 최종 데이터셋(X, y)을 준비합니다."""
    # 사용할 변수 목록 정의
    cand_features = [c for c in ["감정가","최저가","유찰횟수","건물면적","토지면적","건축연수","면적당감정가","최저비율","층","매각연도","매각월"] if c in df.columns]
    categorical_cols = [c for c in ['bjd_sido', 'bjd_sigungu'] if c in df.columns]
    
    # 숫자형 변수 결측치 처리
    X_num = df[cand_features].copy()
    num_imputer = SimpleImputer(strategy="median").fit(X_num)
    X_num = pd.DataFrame(num_imputer.transform(X_num), columns=cand_features, index=df.index)
    
    # 범주형 변수 라벨 인코딩
    X_cat = pd.DataFrame(index=df.index)
    for c in categorical_cols:
        X_cat[c] = df[c].fillna("NA").astype('category').cat.codes
        
    X = pd.concat([X_num, X_cat], axis=1)
    
    # 스케일링
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    # 라벨(y) 생성
    y = df["label"].fillna(0).astype(int)
    
    return X_scaled, y, num_imputer, scaler

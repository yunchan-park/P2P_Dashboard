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
        return R_AVG
    
    if '서울' in location_str or '부산' in location_str:
        return 0.8
    else:
        return 0.7

def calculate_item_specific_r(appraisal_price, min_price, num_failure_rounds):
    """
    개별 경매 건의 감정가, 최저가, 유찰횟수를 기반으로 동적 저감율(r)을 계산합니다.
    계산이 불가능한 경우 (유찰횟수가 0이거나 가격 정보 누락 등) R_AVG를 반환합니다.
    """
    if pd.isna(appraisal_price) or pd.isna(min_price) or num_failure_rounds <= 0 or appraisal_price == 0:
        return R_AVG
    
    try:
        base_ratio = min_price / appraisal_price
        if base_ratio <= 0: return R_AVG

        r_val = base_ratio ** (1 / num_failure_rounds)
        return max(0.01, min(0.99, r_val))
    except (ValueError, OverflowError):
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
    """
    유찰횟수가 있는 모든 경매 건에 대해 과거 유찰 이력을 가상 데이터로 생성하여 증강합니다.
    기본적으로 규칙 기반 저감율을 사용하되, 비현실적인 경우(규칙기반 예측 최저가가 실제 최저가보다 낮은 경우)
    동적 저감율을 계산하여 적용합니다.
    최종 결과물은 (생성된 가상 유찰 데이터) + (원본 낙찰 성공 데이터)로 구성됩니다.
    """
    print(f"[INFO] 원본 데이터 크기: {len(df)}")
    augmented_rows = []

    for _, row in df.iterrows():
        failure_rounds = int(row.get('유찰횟수', 0))
        appraisal_price = row.get('감정가')

        # 1. 증강 대상 선정: 유찰 횟수가 1 이상이고, 감정가가 있는 데이터
        if failure_rounds > 0 and pd.notna(appraisal_price):
            
            # 2. 저감율(r) 결정: 조건부 로직 적용
            rule_based_r = get_rule_based_r(row.get('소재지'))
            # 규칙 기반으로 마지막 유찰 회차의 최저가를 예측해봄
            rule_based_final_min_price = appraisal_price * (rule_based_r ** failure_rounds)
            
            use_r = rule_based_r # 기본적으로 규칙 기반 r 사용
            
            # 규칙 기반 예측이 실제 최저가보다 낮아 비현실적일 경우, 동적 r로 전환
            if rule_based_final_min_price < row.get('최저가', float('inf')):
                use_r = calculate_item_specific_r(appraisal_price, row.get('최저가'), failure_rounds)

            # 3. 가상 유찰 데이터 생성
            for k in range(failure_rounds):
                new_row = row.copy()
                new_row['유찰횟수'] = k
                # 결정된 use_r을 사용하여 최저가 계산
                new_row['최저가'] = appraisal_price * (use_r ** k)
                new_row['label'] = 0  # 가상 데이터는 항상 실패(label=0)
                augmented_rows.append(new_row)

        # 4. 원본 데이터 처리: label이 1인(낙찰 성공) 데이터만 최종 결과에 포함
        if row.get('label') == 1:
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

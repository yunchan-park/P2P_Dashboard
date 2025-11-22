
import joblib
import pandas as pd
import os
import inspect

# City mapping for dynamic model loading
CITY_MAP = {
    '서울': 'seoul',
    '부산': 'busan',
    '대구': 'daegu',
    '인천': 'incheon',
    '광주': 'gwangju',
    '대전': 'daejeon',
    '울산': 'ulsan',
    '경기': 'gyeonggi'
}

# Global variables to store loaded models and encoders to avoid reloading
_loaded_models = {}
_loaded_tes = {}
_loaded_ohes = {}
_loaded_model_columns = {}

_si_code_df = None

# Get the directory of the current script
_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def _load_si_code_df():
    global _si_code_df
    if _si_code_df is None:
        print("Loading si_code.csv...")
        try:
            si_code_path = os.path.join(_script_dir, 'si_code.csv')
            _si_code_df = pd.read_csv(si_code_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"si_code.csv not found at {si_code_path}. Please ensure it exists.")
    return _si_code_df

def _load_city_models(city_korean):
    """Loads models and encoders for a given city, caching them."""
    city_english = None
    for k, v in CITY_MAP.items():
        if city_korean.startswith(k):
            city_english = v
            break
    
    if not city_english:
        raise ValueError(f"Unsupported city: {city_korean}. Please provide a valid city name.")

    if city_english not in _loaded_models:
        print(f"Loading models for {city_korean} ({city_english})...")
        try:
            _loaded_models[city_english] = joblib.load(os.path.join(_script_dir, f'rf_model_{city_english}.joblib'))
            _loaded_tes[city_english] = joblib.load(os.path.join(_script_dir, f'target_encoder_{city_english}.joblib'))
            _loaded_model_columns[city_english] = joblib.load(os.path.join(_script_dir, f'model_columns_{city_english}.joblib'))
            
            # Load OneHotEncoder for cities that use it
            if city_english in ['incheon', 'seoul']:
                _loaded_ohes[city_english] = joblib.load(os.path.join(_script_dir, f'onehot_encoder_{city_english}.joblib'))
            else:
                _loaded_ohes[city_english] = None # Explicitly set to None for other cities
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files for {city_korean} not found. Please ensure {e.filename} exists in the current directory.")
    
    return _loaded_models[city_english], _loaded_tes[city_english], _loaded_ohes[city_english], _loaded_model_columns[city_english]

def predict_price(input_data):
    """
    아파트 정보를 입력받아 예상 매매가를 예측하는 함수
    :param input_data: dict 형태의 아파트 정보
    :return: float 형태의 예측 가격 (만원 단위)
    """
    # Load si_code_df if not already loaded
    si_code_df = _load_si_code_df()

    # --- 1. 입력 데이터 변환 ---
    # 모델이 학습한 형태로 입력 데이터를 재구성합니다.
    sigungu_str = f"{input_data.get('시/도', '')} {input_data.get('구', '')} {input_data.get('동', '')}".strip()
    
    model_input = {
        '시군구': sigungu_str,
        '본번': input_data.get('본번'),
        '부번': input_data.get('부번'),
        '단지명': input_data.get('아파트명'),
        '전용면적(㎡)': input_data.get('전용면적'),
        '계약년월': input_data.get('계약년월'),
        '층': input_data.get('층'),
        '건축년도': input_data.get('건축년도'),
        '가계대출_금리': input_data.get('가계대출_금리')
    }

    # Derive '계약년' and '계약월'
    contract_year_month = model_input.get('계약년월')
    if contract_year_month:
        model_input['계약년'] = contract_year_month // 100
        model_input['계약월'] = contract_year_month % 100
    else:
        model_input['계약년'] = None
        model_input['계약월'] = None

    # Derive '면적구간'
    area = model_input.get('전용면적(㎡)')
    if area is not None:
        if area < 60:
            model_input['면적구간'] = '40~60'
        elif 60 <= area < 85:
            model_input['면적구간'] = '60~85'
        elif 85 <= area < 100:
            model_input['면적구간'] = '85~100'
        elif 100 <= area < 135:
            model_input['면적구간'] = '100~135'
        else:
            model_input['면적구간'] = '135~'
    else:
        model_input['면적구간'] = None

    # 법정동코드 동적으로 찾기
    sido = input_data.get('시/도', '').strip()
    gu = input_data.get('구', '').strip()
    dong = input_data.get('동', '').strip()

    if not (sido and gu and dong):
        raise ValueError("시/도, 구, 동 정보는 법정동코드 조회를 위해 필수입니다.")

    full_dong_name = f"{sido} {gu} {dong}"
    
    # '폐지여부'가 '존재'하는 법정동만 필터링
    filtered_si_code = si_code_df[(si_code_df['법정동명'] == full_dong_name) & (si_code_df['폐지여부'] == '존재')]

    if filtered_si_code.empty:
        raise ValueError(f"'{full_dong_name}'에 해당하는 법정동코드를 찾을 수 없거나 폐지된 지역입니다. 입력 정보를 확인해주세요.")
    elif len(filtered_si_code) > 1:
        # 여러 개가 검색될 경우, 더 구체적인 정보가 필요할 수 있으나, 현재는 첫 번째 항목 사용
        # 실제 데이터에 따라 이 부분은 더 정교하게 처리될 수 있습니다.
        print(f"경고: '{full_dong_name}'에 대해 여러 법정동코드가 발견되었습니다. 첫 번째 코드를 사용합니다.")
        model_input['법정동코드'] = filtered_si_code.iloc[0]['법정동코드']
    else:
        model_input['법정동코드'] = filtered_si_code.iloc[0]['법정동코드']

    # 필수 입력값 확인 (법정동코드는 이제 동적으로 채워지므로 제외)
    required_fields = ['시군구', '단지명', '전용면적(㎡)', '계약년월', '층', '건축년도', '가계대출_금리']
    for field in required_fields:
        if model_input[field] is None:
            # '본번', '부번'은 없을 수 있으므로 예외 처리
            if field not in ['본번', '부번']:
                 raise ValueError(f"필수 입력값이 누락되었습니다: {field}")

    # --- 2. 도시별 모델 및 인코더 로드 ---
    city_korean = input_data.get('시/도', '').strip()
    if not city_korean:
        raise ValueError("입력 데이터에 '시/도' 필드가 반드시 포함되어야 합니다.")
        
    model, te, ohe, model_columns = _load_city_models(city_korean)

    # --- 3. 예측을 위한 데이터프레임 생성 ---
    df = pd.DataFrame([model_input])

    # --- 4. 인코딩 적용 ---
    # Target Encoding
    df_encoded = te.transform(df)
    
    # One-Hot Encoding (해당 도시 모델이 OHE를 사용하는 경우)
    if ohe:
        df_encoded = ohe.transform(df_encoded)

    # --- 5. 컬럼 정렬 ---
    # 학습 시점의 컬럼 순서와 동일하게 맞추고, 없는 컬럼은 0으로 채웁니다.
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)

    # --- 6. 가격 예측 ---
    prediction = model.predict(df_aligned)

    return prediction[0]

if __name__ == '__main__':
    # --- 예측할 아파트 정보 입력 ---
    # 사용자가 제공한 형식에 따른 예시 데이터
    sample_data = {
        #--- 사용자 입력 ---
        '시/도': '서울특별시',
        '구': '강남구',
        '동': '개포동',
        '본번': 658,
        '부번': 0,
        '아파트명': '개포6차우성아파트1동~8동',
        '전용면적': 167.1,
        '건축년도': 1987,
        #--- 다른 모델에서 사용될 수 있는 변수 ---
        '동(선택)': '101동',
        '호수(선택)': '101호',
        '감정가격(원)': 2500000000,
        '선순위채권 가격(원)': 500000000,
        #--- 예측을 위해 추가로 필요한 변수 (사용자가 제공해야 함) ---
        '층': 3,
        '계약년월': 202501, # 예측하고 싶은 시점 (YYYYMM 형식)
        '가계대출_금리': 3.5 # 예측 시점의 금리
    }

    try:
        # 함수 호출
        predicted_price = predict_price(sample_data)

        # 결과 출력
        print("--- 입력된 아파트 정보 ---")
        for key, value in sample_data.items():
            print(f"- {key}: {value}")
        print("---" * 10)
        
        # 억 단위로 변환하여 보기 쉽게 출력
        predicted_price_eok = int(predicted_price / 10000)
        predicted_price_man = int(predicted_price % 10000)
        print(f"==> 예상 매매가: 약 {predicted_price_eok}억 {predicted_price_man}만 원")

    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] 예측 중 오류가 발생했습니다: {e}")

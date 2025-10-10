import pandas as pd
import re
import os

# --- 파일 경로 설정 ---
data_dir = '/Users/eoseungyun/Desktop/학교/3-2/데캡디/모델링/Modeling/Auction/Data'
auction_file = os.path.join(data_dir, 'auction.csv')
code_file = os.path.join(data_dir, 'si_code.csv')
sales_file = os.path.join(data_dir, 'sales.xlsx')
output_processed_file = os.path.join(data_dir, 'auction_processed.csv')
output_final_file = os.path.join(data_dir, 'auction_preprocessed.csv')

# --- 1. 데이터 및 조회용 데이터 불러오기 ---
try:
    auction_df = pd.read_csv(auction_file)
    code_df = pd.read_csv(code_file)
    sales_df = pd.read_excel(sales_file)
    print("모든 소스 파일을 성공적으로 불러왔습니다.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요. - {e}")
    exit()

# --- 2. 조회용 데이터 준비 ---
valid_districts = set(code_df[code_df['폐지여부'] == '존재']['법정동명'].unique())
sales_df.dropna(subset=['도로명', '시군구', '번지'], inplace=True)
sales_df['도로명_키'] = sales_df['도로명'].str.strip().str.replace(' ', '')
address_lookup = {k: v for k, v in zip(sales_df['도로명_키'], sales_df[['시군구', '번지']].to_dict('records'))}
print(f"{len(valid_districts)}개의 법정동명, {len(address_lookup)}개의 도로명-지번 조회 데이터를 준비했습니다.")

# --- 3. Helper 함수 정의 ---
def classify_address_type(location_str, districts_set):
    if not isinstance(location_str, str): return '알 수 없음'
    first_three_words = " ".join(location_str.split()[:3])
    return '지번 주소' if first_three_words in districts_set else '도로명 주소'

def parse_lot_based(location_str):
    parts = location_str.split()
    sigungu_parts = []
    for part in parts:
        if part.endswith(('시', '도', '구', '군', '동', '리', '가')):
            sigungu_parts.append(part)
        else:
            break
    sigungu = " ".join(sigungu_parts)
    remaining_parts = parts[len(sigungu_parts):]
    bunji, others = None, " ".join(remaining_parts)
    for i, part in enumerate(remaining_parts):
        if re.match(r'^[0-9,-]+', part):
            bunji = part
            others = " ".join(remaining_parts[i+1:])
            break
    return sigungu, bunji, others

def convert_street_to_lot(location_str, lookup_dict):
    address_core = re.split(r'[,(]', location_str)[0].strip()
    sigungu_match = re.match(r'\S+\s\S+', address_core)
    if not sigungu_match: return None
    road_address_part = address_core[sigungu_match.end():].strip()
    lookup_key = road_address_part.replace(' ', '')
    found_data = lookup_dict.get(lookup_key)
    if found_data:
        sigungu = found_data['시군구']
        bunji = str(found_data['번지'])
        others = location_str.split(address_core)[-1].strip(', ').strip()
        return sigungu, bunji, others
    else:
        return None

def process_row(row):
    location = row['location']
    addr_type = classify_address_type(location, valid_districts)
    if addr_type == '도로명 주소':
        result = convert_street_to_lot(location, address_lookup)
        if result is not None:
            return result
    return parse_lot_based(location)

# --- 4. 주소 전처리 실행 ---
print("\n1단계: 주소 전처리를 시작합니다...")
auction_df[['시군구', '번지', '이외']] = auction_df.apply(process_row, axis=1, result_type='expand')
auction_df.to_csv(output_processed_file, index=False, encoding='utf-8-sig')
print(f"1단계 완료: 전처리된 파일이 '{output_processed_file}'에 저장되었습니다.")

# --- 5. 법정동 코드 매핑 ---
print("\n2단계: 법정동 코드 매핑을 시작합니다...")
code_df_active = code_df[code_df['폐지여부'] == '존재'][['법정동명', '법정동코드']]
final_df = pd.merge(auction_df, code_df_active, left_on='시군구', right_on='법정동명', how='left')

failed_count = final_df['법정동코드'].isnull().sum()
print(f"매핑에 실패한 데이터 수: {failed_count}")
if '법정동명' in final_df.columns:
    final_df.drop(columns=['법정동명'], inplace=True)

# --- 6. 최종 결과 저장 및 출력 ---
final_df.to_csv(output_final_file, index=False, encoding='utf-8-sig')
print(f"2단계 완료: 법정동 코드가 추가된 최종 파일을 '{output_final_file}'로 저장했습니다.")

print("\n--- 최종 데이터 샘플 ---")
print(final_df[['시군구', '번지', '법정동코드']].head().to_string())
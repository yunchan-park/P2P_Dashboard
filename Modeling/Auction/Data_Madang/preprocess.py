import pandas as pd
import re
import os
import math
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --- 파일 경로 ---
data_dir = os.path.dirname(os.path.abspath(__file__))
sales_file = os.path.join(data_dir, 'sales_data_combined.csv')
auction_file = os.path.join(data_dir, 'auction_data_combined.csv')
si_code_file = os.path.join(data_dir, 'si_code.csv')
output_file = os.path.join(data_dir, 'auction_preprocessed.csv')

# --- 1. 조회 딕셔너리 생성 함수들 ---

def create_road_address_lookup(sales_df):
    print("도로명 주소용 조회 딕셔너리를 생성합니다...")
    
    lookup_df = sales_df.dropna(subset=['도로명', '건축년도', '시군구', '본번', '부번']).copy()
    lookup_df['조회키'] = lookup_df['도로명'].apply(lambda x: x.replace(' ', '') if isinstance(x, str) else None)
    lookup_df.dropna(subset=['조회키'], inplace=True)

    final_lookup = {}
    
    grouped = lookup_df.groupby('조회키').agg({
        '건축년도': lambda x: list(x.unique()),
        '시군구': lambda x: list(x.unique()),
        '본번': lambda x: list(x.unique()),
        '부번': lambda x: list(x.unique())
    })

    for key, row in grouped.iterrows():
        years = row['건축년도']
        
        final_year = math.ceil(sum(years) / len(years)) if len(years) > 1 else years[0]
        final_sigungu = row['시군구'][0] if len(row['시군구']) == 1 else pd.Series(row['시군구']).mode()[0]
        final_bonbeon = row['본번'][0] if len(row['본번']) == 1 else pd.Series(row['본번']).mode()[0]
        final_bubeon = row['부번'][0] if len(row['부번']) == 1 else pd.Series(row['부번']).mode()[0]

        final_lookup[key] = {
            '건축년도': final_year,
            '시군구': final_sigungu,
            '본번': final_bonbeon,
            '부번': final_bubeon
        }
        
    print(f"총 {len(final_lookup)}개의 도로명 키 생성 완료.")
    return final_lookup

def create_lot_address_lookup(sales_df):
    print("지번 주소용 조회 딕셔너리를 생성합니다...")
    
    lookup_df = sales_df.dropna(subset=['시군구', '번지', '건축년도', '본번', '부번']).copy()
    
    grouped = lookup_df.groupby(['시군구', '번지']).agg({
        '건축년도': lambda x: x.mode()[0] if not x.mode().empty else None,
        '본번': lambda x: x.mode()[0] if not x.mode().empty else None,
        '부번': lambda x: x.mode()[0] if not x.mode().empty else None
    }).reset_index()
    
    final_lookup = {}
    for _, row in grouped.iterrows():
        key = (row['시군구'], str(row['번지']))
        final_lookup[key] = {
            '건축년도': row['건축년도'],
            '본번': row['본번'],
            '부번': row['부번']
        }
        
    print(f"총 {len(final_lookup)}개의 지번 키 생성 완료.")
    return final_lookup

# --- 2. 경매 데이터 파싱 함수들 ---

def classify_address_type(location_str):
    if isinstance(location_str, str) and re.search(r'\s\S+(?:대로|로|길)\s', location_str):
        return '도로명 주소'
    return '지번 주소'

def extract_floor_from_string(text_part):
    if not isinstance(text_part, str): return 0
    floor_match = re.search(r'(\d+)\s*층', text_part)
    if floor_match: return int(floor_match.group(1))
    unit_matches = re.findall(r'(\d+)\s*호', text_part)
    if unit_matches:
        last_unit_str = unit_matches[-1]
        try:
            unit_num = int(last_unit_str)
            if unit_num >= 100:
                floor_str = last_unit_str[:-2]
                if floor_str: return int(floor_str)
        except ValueError: pass
    return 0

def parse_full_address(location_str):
    sigungu, bunji, road_addr, remainder = None, None, None, location_str
    addr_type = classify_address_type(location_str)
    words = location_str.split()
    
    if addr_type == '도로명 주소':
        for i, word in enumerate(words):
            if word.endswith(('로', '길', '대로')) and not word.isdigit():
                sigungu = " ".join(words[:i])
                road_addr_parts = [word]
                remainder_index = i + 1
                if (i + 1) < len(words):
                    candidate = re.split(r'[,(]', words[i+1])[0]
                    if candidate.replace('-', '').isdigit():
                        road_addr_parts.append(candidate)
                        remainder_index = i + 2
                road_addr = " ".join(road_addr_parts)
                remainder = " ".join(words[remainder_index:])
                break
    else: # 지번 주소
        for i, part in enumerate(words):
            if (re.match(r'^\d+-\d+$', part) or re.match(r'^\d+$', part)) and i > 0:
                sigungu = " ".join(words[:i])
                bunji = part
                remainder = " ".join(words[i+1:])
                break
        if not bunji: sigungu = location_str
    
    return sigungu, bunji, road_addr, remainder, addr_type

# --- 3. 메인 실행 로직 ---
def main():
    print("데이터 파일을 로드합니다...")
    sales_df = pd.read_csv(sales_file, encoding='utf-8')
    auction_df = pd.read_csv(auction_file)
    si_code_df = pd.read_csv(si_code_file)
    print("로드 완료.")

    # 1. 두 종류의 조회 딕셔너리 생성
    road_lookup = create_road_address_lookup(sales_df)
    lot_lookup = create_lot_address_lookup(sales_df)

    # 2. 최종 매핑 함수 정의
    def get_all_info(location_str):
        sigungu_parsed, bunji_parsed, road_addr_parsed, remainder, addr_type = parse_full_address(location_str)
        floor = extract_floor_from_string(remainder)
        
        mapped_info = None
        
        if addr_type == '도로명 주소':
            if road_addr_parsed:
                key = road_addr_parsed.replace(' ', '')
                if key in road_lookup:
                    mapped_info = road_lookup[key]
        
        if mapped_info is None and sigungu_parsed and bunji_parsed:
            lot_key = (sigungu_parsed, bunji_parsed)
            if lot_key in lot_lookup:
                mapped_info = lot_lookup[lot_key]
                mapped_info['시군구'] = sigungu_parsed
        
        if mapped_info:
            return pd.Series({
                '건축년도': mapped_info.get('건축년도'),
                '시군구': mapped_info.get('시군구'),
                '본번': mapped_info.get('본번'),
                '부번': mapped_info.get('부번'),
                '층': floor
            })
        else:
            return pd.Series({
                '건축년도': None, '시군구': None,
                '본번': None, '부번': None, '층': floor
            })

    # 3. 전체 데이터에 매핑 적용
    print("\n최종 하이브리드 로직으로 전체 데이터 매핑을 시작합니다...")
    mapped_results_df = auction_df['소재지'].apply(get_all_info)
    
    # 원본 auction_df에서 중복될 수 있는 컬럼 제거 후 병합
    cols_to_drop = [col for col in mapped_results_df.columns if col in auction_df.columns]
    auction_df_reduced = auction_df.drop(columns=cols_to_drop)
    final_df = pd.concat([auction_df_reduced, mapped_results_df], axis=1)

    # 4. 매핑 성공 데이터만 필터링
    preprocessed_df = final_df.dropna(subset=['건축년도']).copy()

    # 5. 법정동코드 매핑 추가
    print("법정동코드를 매핑합니다...")
    si_code_to_merge = si_code_df[['법정동명', '법정동코드']].drop_duplicates()
    # '시군구' 컬럼을 기준으로 병합
    preprocessed_df = pd.merge(preprocessed_df, si_code_to_merge, left_on='시군구', right_on='법정동명', how='left')
    preprocessed_df.drop(columns=['법정동명'], inplace=True) # 중복 컬럼 제거

    # 6. 결과 집계 및 보고
    total_original_count = len(auction_df)
    total_processed_count = len(preprocessed_df)
    success_rate = (total_processed_count / total_original_count) * 100
    
    print("\n--- 최종 매핑 결과 ---")
    print(f"총 원본 데이터: {total_original_count}건")
    print(f"매핑 성공 및 필터링된 데이터: {total_processed_count}건")
    print(f"최종 데이터 성공률: {success_rate:.2f}%")

    # 7. 결과 파일 저장
    preprocessed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n최종 전처리된 데이터를 '{output_file}'에 저장했습니다.")

if __name__ == '__main__':
    main()
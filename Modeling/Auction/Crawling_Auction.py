from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import re
import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import itertools

# 조건 설정 및 데이터 크롤링
def setting_and_crawl(driver, auction_list, loc, from_date, to_date, bid_result, asset_type):
    auction_site = 'https://www.onbid.co.kr/op/bda/bidrslt/collateralRealEstateBidResultList.do'
    driver.get(auction_site)
    
    try:
        # 안정적인 로딩을 위해 주요 요소가 나타날 때까지 대기
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "dpslMtd2")))
        
        # 처분방식 설정 (매각)
        sale_button = driver.find_element(By.ID, "dpslMtd2")
        sale_button.click()
        print("매각 완료")

        # 용도 설정(주거용 건물 - 아파트)
        select = Select(driver.find_element(By.ID, "searchCtgrId2"))
        select.select_by_visible_text("주거용건물")
        print("주거용 완료")

        select = Select(driver.find_element(By.ID, "searchCtgrId"))
        select.select_by_visible_text("아파트")
        print("아파트 완료")

        # 지역 설정
        select = Select(driver.find_element(By.ID, "siDo"))
        select.select_by_visible_text(loc)
        print(f"{loc} 완료")

        # 날짜 설정
        date_from_input = driver.find_element(By.ID, "searchBidDateFrom")
        date_from_input.clear()
        date_from_input.send_keys(from_date)
        print(f"시작 날짜({from_date}) 완료")

        date_to_input = driver.find_element(By.ID, "searchBidDateTo")
        date_to_input.clear()
        date_to_input.send_keys(to_date)
        print(f"종료 날짜({to_date}) 완료")

        # 입찰 결과 설정
        select = Select(driver.find_element(By.ID, "searchPbctStatCd"))
        select.select_by_visible_text(bid_result)
        print(f"입찰 결과({bid_result}) 완료")

        # 자산 구분 설정
        select = Select(driver.find_element(By.ID, "searchPrptDvsnCd"))
        select.select_by_visible_text(asset_type)
        print(f"자산 구분({asset_type}) 완료")

        # 100줄씩 보기
        select = Select(driver.find_element(By.ID, "pageUnit"))
        select.select_by_visible_text("100줄씩 보기")
        print("정렬 완료")

        # 검색 버튼 클릭
        driver.find_element(By.ID, "searchBtn").click()
        print("검색 시작")

        # 검색 결과 테이블이 로드될 때까지 대기
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "op_tbl_type1")))
        
        # 경매 정보 불러오기
        auction_info(driver, auction_list)

    except TimeoutException:
        print("페이지 로딩 시간 초과. 다음 조건으로 넘어갑니다.")
    except Exception as e:
        print(f"조건 설정 또는 검색 중 오류 발생: {e}")

# 경매 정보 불러오기
def auction_info(driver, auction_list):
    try:
        # 결과 테이블의 tbody가 없을 경우(검색 결과 없음) 처리
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".op_tbl_type1 tbody")))
        result_table_body = driver.find_element(By.CSS_SELECTOR, ".op_tbl_type1 tbody")
        
        # 초기 행 목록을 가져와서 개수만 확인
        initial_auctions = result_table_body.find_elements(By.TAG_NAME, 'tr')
        num_auctions = len(initial_auctions)

        if num_auctions == 0 or (num_auctions == 1 and "없습니다" in initial_auctions[0].text):
            print("검색 결과가 없습니다.")
            return

        print(f"{num_auctions}개의 경매 정보를 가져옵니다.")
        
        for i in range(num_auctions):
            try:
                # StaleElementReferenceException 방지를 위해 매번 요소를 다시 찾음
                all_rows = driver.find_elements(By.CSS_SELECTOR, ".op_tbl_type1 tbody tr")
                if i >= len(all_rows):
                    print(f"행 {i}를 다시 찾을 수 없습니다. 다음으로 넘어갑니다.")
                    continue
                auction = all_rows[i]

                # 물건정보(위치, 토지 크기, 건물 크기)
                building_info = auction.find_element(By.CLASS_NAME, 'al')
                location = building_info.find_element(By.CSS_SELECTOR, "dd.fwb").text
                
                area = building_info.find_elements(By.TAG_NAME, "dd")[-1].text
                land_area_match = re.search(r"토지\s*([\d,.]+)", area)
                land_area = land_area_match.group(1).replace(',', '') if land_area_match else "0"
                apt_area_match = re.search(r"건물\s*([\d,.]+)", area)
                apt_area = apt_area_match.group(1).replace(',', '') if apt_area_match else "0"
                
                # 가격 정보 (최저 입찰가, 낙찰가)
                price_elements = auction.find_elements(By.CLASS_NAME, 'ar')
                min_price = price_elements[0].text.replace(',', '')
                win_price = price_elements[1].text.replace(',', '')
                
                # 이외 정보 (입찰 결과, 개찰 일시)
                info_elements = auction.find_elements(By.TAG_NAME, 'td')
                result = info_elements[4].text
                result_time = info_elements[5].text
            
                auction_list.append({
                    'location': location,
                    'landarea': land_area,
                    'aptarea': apt_area,
                    'minprice': min_price,
                    'winprice': win_price,
                    'result': result,
                    'time': result_time
                })
                
            except Exception as e:
                print(f"개별 경매 정보(행 {i}) 처리 중 오류: {e}")

    except NoSuchElementException:
        print("검색 결과 테이블을 찾을 수 없습니다. (결과 없음)")
    except TimeoutException:
        print("결과 테이블 로딩 시간 초과.")
    except Exception as e:
        print(f"경매 정보 불러오기 중 오류: {e}")


# 검색 조건 list 생성
locs = ["서울특별시"]
dates = [["2020-01-01", "2021-01-01"],
         ["2021-01-02", "2022-01-01"], 
         ["2022-01-02", "2023-01-01"], 
         ["2023-01-02", "2024-01-01"], 
         ["2024-01-02", "2025-01-01"], 
         ["2025-01-02", "2026-01-01"]]
bid_results = ["낙찰", "유찰"]
asset_types = ["기타일반재산", "금융권담보재산"]

combinations = list(itertools.product(locs, dates, bid_results, asset_types))

# 크롤링 시작
chromedriver_path = './chromedriver'
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service)

auction_data_list = []
try:
    for i, (loc, date_range, bid_result, asset_type) in enumerate(combinations):
        print(f"--- {i+1}/{len(combinations)} 번째 조건 크롤링 시작 ---")
        from_date = date_range[0]
        to_date = date_range[1]
        
        setting_and_crawl(driver, auction_data_list, loc, from_date, to_date, bid_result, asset_type)
        
        # 서버 부하를 줄이기 위해 잠시 대기
        time.sleep(2)

finally:
    driver.quit()
    print("--- 크롤링 완료 ---")

    # 데이터프레임 생성 및 엑셀 저장
    if auction_data_list:
        df = pd.DataFrame(auction_data_list)
        excel_filename = "auction.xlsx"
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"{len(df)}개의 데이터를 '{excel_filename}' 파일로 저장했습니다.")
    else:
        print("수집된 데이터가 없어 파일을 저장하지 않았습니다.")
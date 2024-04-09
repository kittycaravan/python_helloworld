import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

####    상수들  ################
SIZE_ASC = 8
SIZE_DESC = 4
TARGET_MONTH = 5
################################

# 오름차순용 배열 생성 (2차원 배열)
array_asc = np.arange(1, SIZE_ASC + 1).reshape(-1, 1)
# 내림차순용 배열 생성 (2차원 배열)
array_desc = np.arange(1, SIZE_DESC + 1).reshape(-1, 1)

# 둘 합친거
array_combine = np.arange(1, SIZE_ASC+SIZE_DESC + 1).reshape(-1, 1)

# 작년의 평균 최저 기온 월부터 최고 기온 월까지 각 월의 평균 기온 값을 넣은 배열 생성 (데이터 기준 1월부터 8월까지, 8개월)
last_year_month_temp_asc = np.array([
    [-1.5, 2.3, 9.8, 13.8, 19.5, 23.4, 26.7, 27.2]
]).reshape(-1, 1)

# 작년의 평균 최고 기온 월부터 최저 기온 월까지 각 월의 평균 기온 값을 넣은 배열 생성 (데이터 기준 8월부터 1월까지, 6개월)
last_year_month_temp_desc = np.array([
    [23.7, 15.8, 6.8, 1.1]
]).reshape(-1, 1)

# 두 배열 합치기
last_year_month_temp_combined = np.concatenate((last_year_month_temp_asc, last_year_month_temp_desc), axis=0)
print("Combined Array:" , last_year_month_temp_combined)

# 선형 회귀 모델 초기화
model_asc = LinearRegression()
model_desc = LinearRegression()

# 선형 회귀 모델에 데이터 학습
model_asc.fit(array_asc, last_year_month_temp_asc)
model_desc.fit(array_desc, last_year_month_temp_desc)

## 모델을 사용하여 특정 월의 날씨 예측
# 월 분기처리
if(TARGET_MONTH <= 8):
    target_day = np.array([[TARGET_MONTH]]) # 예상 월 및 일 (소수점으로 일을 지정)
    predicted_temperature = model_asc.predict(target_day)
    print("예상 평균 온도 (상승):", predicted_temperature[0][0])
else:
    target_day = np.array([[TARGET_MONTH-8]]) # 예상 월 및 일 (소수점으로 일을 지정)
    predicted_temperature = model_desc.predict(target_day)
    print("예상 평균 온도 (하강):", predicted_temperature[0][0])

# 그래프 그리기
plt.rcParams['font.family'] = 'Malgun Gothic'   #한글깨지면 이거 하면 됨. 폰트 지정. utf8문제가 아님.
plt.figure(figsize=(13, 6)) #그래프 전체 크기 (수치들 말고 전체 화면 크기. 인치단위)
plt.xticks(np.arange(1, SIZE_ASC+SIZE_DESC+1)) # 1부터 12까지의 눈금을 표시합니다.

## 월별 평균기온 그래프
# 'bo-'는 그래프에서 사용하는 선의 스타일을 지정하는 것입니다. 
# 여기서 'b'는 파란색(blue)을 나타내는 색상을 의미하고, 'o'는 원(circle)을 의미합니다. 
# 마지막으로 '-'는 선으로 연결된 점들을 의미합니다. 따라서 'bo-'는 파란색 원으로 표시되는 데이터 포인트들을 선으로 연결하여 그려라는 의미입니다.
# 이 그래프에서 'bo-'는 평균 온도 변화를 나타내는 데이터 포인트들을 파란색 원으로 표시하고, 그 원들을 선으로 연결하여 그래프를 그리라는 지시입니다.
plt.plot(array_combine, last_year_month_temp_combined, 'bo-', label='평균온도 변화')

## 예측 월 그래프(쩜)
# 월 분기처리
if(TARGET_MONTH <= 8):
    plt.plot(target_day, predicted_temperature, 'ro', label='예측 평균온도')
else:
    plt.plot(target_day+8, predicted_temperature, 'ro', label='예측 평균온도')

plt.xlabel('월')
plt.ylabel('온도')
plt.title('과거 월별 평균 기온과 예상 월 비교')
plt.legend()
plt.grid(True)
plt.show()
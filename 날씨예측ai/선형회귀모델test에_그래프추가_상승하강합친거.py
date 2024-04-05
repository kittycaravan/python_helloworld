import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

SIZE_ASC = 8
SIZE_DESC = 5

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
# last_year_month_temp_desc = np.array([
#     [27.2, 23.7, 15.8, 6.8, 1.1, -1.5]
# ]).reshape(-1, 1)
last_year_month_temp_desc = np.array([
    [23.7, 15.8, 6.8, 1.1, -1.5]
]).reshape(-1, 1)

# 두 배열 합치기
last_year_month_temp_combined = np.concatenate((last_year_month_temp_asc, last_year_month_temp_desc), axis=0)

print("Combined Array:")
print(last_year_month_temp_combined)


# 선형 회귀 모델 초기화
model_asc = LinearRegression()
model_desc = LinearRegression()

# 선형 회귀 모델에 데이터 학습
model_asc.fit(array_asc, last_year_month_temp_asc)
model_desc.fit(array_desc, last_year_month_temp_desc)

# 예상 월 및 일 (소수점으로 일을 지정)
target_day = np.array([[7.5]])

# 모델을 사용하여 올해 1월 1일의 날씨 예측
predicted_temperature_asc = model_asc.predict(target_day)
predicted_temperature_desc = model_desc.predict(target_day)

print("예상 평균 온도 (상승):", predicted_temperature_asc[0][0])
print("예상 평균 온도 (하강):", predicted_temperature_desc[0][0])

# 그래프 그리기
# plt.figure(figsize=(8, 6))
# plt.figure(figsize=(14, 6))
plt.figure(figsize=(13, 6))

# 상승 그래프
plt.plot(array_combine, last_year_month_temp_combined, 'bo-', label='Combine Months')
# plt.plot(target_day, predicted_temperature_asc, 'ro', label='Predicted Ascending Temperature')

# 하강 그래프

plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Average Monthly Temperature')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

SIZE_ASC = 8
SIZE_DESC = 6

# 오름월용 배열 생성 (2차원배열로)
array_asc = np.arange(1, SIZE_ASC + 1).reshape(-1, 1)
# 내림월용 배열 생성 (2차원배열로)
array_desc = np.arange(1, SIZE_DESC + 1).reshape(-1, 1)

# 작년의 평균 최저기온 월부터 최고 기온 월까지 각 월의 평균기온 값을 넣은 배열 생성(데이터 기준 1월~8월. 8개월)
last_year_month_temp_asc = np.array([
    [-1.5, 2.3, 9.8, 13.8, 19.5, 23.4, 26.7, 27.2]
]).reshape(-1, 1)

# 작년의 평균 최고기온 월부터 최저 기온 월까지 각 월의 평균기온 값을 넣은 배열 생성(데이터 기준 8월~1월. 6개월)
last_year_month_temp_desc = np.array([
    [27.2, 23.7, 15.8, 6.8, 1.1, -1.5]
]).reshape(-1, 1)

# 선형 회귀 모델 초기화
model_asc = LinearRegression()
model_desc = LinearRegression()

# 선형 회귀 모델에 데이터 학습
model_asc.fit(array_asc, last_year_month_temp_asc)
model_desc.fit(array_desc, last_year_month_temp_desc)

# 예상할 월 (소수점으로 일을 고르는식으로. todo: 각 일도 정확하게 대입가능하게 )
target_day = np.array([[7.5]])

# 모델을 사용하여 올해 1월 1일의 날씨 예측
predicted_temperature_asc = model_asc.predict(target_day)
predicted_temperature_desc = model_desc.predict(target_day)

print("예상 평균 온도(오름차순):", predicted_temperature_asc[0][0])
print("예상 평균 온도(내림차순):", predicted_temperature_desc[0][0])

# 그래프 그리기
plt.figure(figsize=(8, 6))

# 오름차순 그래프
plt.plot(array_asc, last_year_month_temp_asc, 'bo-', label='Ascending Months')
plt.plot(target_day, predicted_temperature_asc, 'ro', label='Predicted Ascending Temperature')

# 내림차순 그래프
plt.plot(array_desc, last_year_month_temp_desc, 'go-', label='Descending Months')
plt.plot(target_day, predicted_temperature_desc, 'yo', label='Predicted Descending Temperature')

plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Average Monthly Temperature')
plt.legend()
plt.grid(True)
plt.show()

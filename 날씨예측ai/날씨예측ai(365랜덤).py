import numpy as np
from sklearn.linear_model import LinearRegression

# 작년 한 해의 평균 기온 데이터(임의의 값)
avg_temperatures_last_year = np.random.randint(low=-10, high=30, size=365).reshape(-1, 1)

# 작년의 날짜를 기준으로 1부터 365까지의 순서로 날짜 배열 생성
dates_last_year = np.arange(1, 366).reshape(-1, 1)

# 선형 회귀 모델 초기화
model = LinearRegression()

# 선형 회귀 모델에 데이터 학습
model.fit(dates_last_year, avg_temperatures_last_year)

# 올해 1월 1일의 날짜 데이터(366일째)
jan_1_date_this_year = np.array([[366]])

# 모델을 사용하여 올해 1월 1일의 날씨 예측
predicted_temperature_jan_1 = model.predict(jan_1_date_this_year)

# 소수점 이하 자리를 한 자리로 제한
predicted_temperature_jan_1_rounded = round(predicted_temperature_jan_1[0][0], 1)

print("올해 1월 1일의 예상 평균 기온:", predicted_temperature_jan_1_rounded, "도")

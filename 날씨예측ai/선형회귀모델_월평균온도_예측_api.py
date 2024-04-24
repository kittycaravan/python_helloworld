import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#request 는 get방식으로 파라미터 받아오려고 추가함
from flask import Flask, jsonify, request

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)


####    상수들  ################
SIZE_ASC = 8
SIZE_DESC = 4
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

# '/get_temp' 엔드포인트에 대한 GET 요청을 처리하는 함수를 정의합니다.

@app.route('/get_temp', methods=['GET'])
def get_temp(): 
    target_month = int(request.args.get('m')) #파라미터로 넘어온 값을 빼서 변수에 대입함.
    if(target_month <= 8):
        target_day = np.array([[target_month]]) # 예상 월 및 일 (소수점으로 일을 지정)
        predicted_temperature = model_asc.predict(target_day)
        print("예상 평균 온도 (상승):", predicted_temperature[0][0])
    else:
        target_day = np.array([[target_month-8]]) # 예상 월 및 일 (소수점으로 일을 지정)
        predicted_temperature = model_desc.predict(target_day)
        print("예상 평균 온도 (하강):", predicted_temperature[0][0])
    predicted_temperature_rounded = round(predicted_temperature[0][0], 1)
    json_data = {"result": predicted_temperature_rounded}
    return jsonify(json_data)  

#### 아래 코드는 pj 폴더 내에 한군데만 있어야함.
# 이 코드를 직접 실행할 때만 Flask 서버를 실행합니다.
# 다른 파일에서 이 코드를 import할 때는 실행되지 않습니다.

if __name__ == '__main__':
    # Flask 애플리케이션을 디버그 모드로 실행합니다.
    # 디버그 모드를 사용하면 코드를 수정하고 저장할 때마다 서버가 자동으로 재시작됩니다.

    app.run(debug=True)
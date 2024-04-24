import numpy as np
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression

print("==== ==== API 서버 런! v0.0.6 ==== ====")
app = Flask(__name__)
sizeAsc = 8
sizeDesc = 4
arrayAsc = np.arange(1, sizeAsc + 1).reshape(-1, 1)
arrayDesc = np.arange(1, sizeDesc + 1).reshape(-1, 1)
arrayCombine = np.arange(1, sizeAsc+sizeDesc + 1).reshape(-1, 1)
lastYearMonthTempAsc = np.array([[-1.5, 2.3, 9.8, 13.8, 19.5, 23.4, 26.7, 27.2]]).reshape(-1, 1)
lastYearMonthTempDesc = np.array([[23.7, 15.8, 6.8, 1.1]]).reshape(-1, 1)
lastYearMonthTemp = np.concatenate((lastYearMonthTempAsc, lastYearMonthTempDesc), axis=0)

#모델 초기화 및 데이터 학습
modelAsc = LinearRegression()
modelDesc = LinearRegression()
modelAsc.fit(arrayAsc, lastYearMonthTempAsc)
modelDesc.fit(arrayDesc, lastYearMonthTempDesc)

@app.route('/get_temp', methods=['GET'])
def get_temp(): 
    targetMonth = int(request.args.get('m'))
    if(targetMonth <= 8):
        targetDay = np.array([[targetMonth]])
        predictTemp = modelAsc.predict(targetDay)
    else:
        targetDay = np.array([[targetMonth-8]])
        predictTemp = modelDesc.predict(targetDay)
    print("======== 예상 온도:", predictTemp[0][0])
    predictTempRound = round(predictTemp[0][0], 1)
    jsonData = {"result": predictTempRound}
    return jsonify(jsonData)  

#서버 런
if __name__ == '__main__':
    app.run(debug=True)
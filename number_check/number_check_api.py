import cv2
import numpy as np
import tensorflow as tf

#request 는 get방식으로 파라미터 받아오려고 추가함
# from flask import Flask, jsonify, request
from flask import Flask, jsonify

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)


#### 0.MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#### 1.데이터 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

#### 2.모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

#### 3.모델 컴파일
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#### 4.모델 훈련
model.fit(train_images, train_labels, epochs=5)


#####################################################
@app.route('/get_number', methods=['GET'])
def get_temp(): 


    #### 5.예측을 위해 올려둔 이미지 파일 읽기
    img = cv2.imread('7.bmp') # 여기 바꿔가면서 테스트하자. 7 정답

    #### 6.이미지 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (28, 28))
    img_processed = np.array(resized, dtype=np.float32).reshape(1, 28, 28) / 255.0

    #### 7.숫자 예측
    prediction = model.predict(img_processed)
    predicted_class = np.argmax(prediction)

    #### 8.예측 결과 출력
    print("이미지에 있는 숫자는:", predicted_class)


    # json_data = {"result": predicted_class}
    json_data = {"result": str(predicted_class)}
    return jsonify(json_data)  

#### 아래 코드는 pj 폴더 내에 한군데만 있어야함.
# 이 코드를 직접 실행할 때만 Flask 서버를 실행합니다.
# 다른 파일에서 이 코드를 import할 때는 실행되지 않습니다.
if __name__ == '__main__':
    # Flask 애플리케이션을 디버그 모드로 실행합니다.
    # 디버그 모드를 사용하면 코드를 수정하고 저장할 때마다 서버가 자동으로 재시작됩니다.
    app.run(debug=True)




import cv2
import numpy as np
import tensorflow as tf

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=5)

# 이미지 읽기
img = cv2.imread('0.bmp') # 여기 바꿔가면서 테스트하자. 5 7 오답계속
# img = cv2.imread('1.bmp') # 여기 바꿔가면서 테스트하자. 1 정답
# img = cv2.imread('2.bmp') # 여기 바꿔가면서 테스트하자. 2 정답
# img = cv2.imread('3.bmp') # 여기 바꿔가면서 테스트하자. 3 정답
# img = cv2.imread('4.bmp') # 여기 바꿔가면서 테스트하자. 4 정답
# img = cv2.imread('5.bmp') # 여기 바꿔가면서 테스트하자. 5 정답
# img = cv2.imread('6.bmp') # 여기 바꿔가면서 테스트하자. 4 5 오답계속
# img = cv2.imread('7.bmp') # 여기 바꿔가면서 테스트하자. 7 정답
# img = cv2.imread('8.bmp') # 여기 바꿔가면서 테스트하자. 8 정답
# img = cv2.imread('9.bmp') # 여기 바꿔가면서 테스트하자. 9 정답

if img is None:
    print("이미지를 읽을 수 없습니다.")
else:
    # 이미지 전처리 및 예측 수행

    # 이미지 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (28, 28))
    img_processed = np.array(resized, dtype=np.float32).reshape(1, 28, 28) / 255.0

    # 숫자 예측
    prediction = model.predict(img_processed)
    predicted_class = np.argmax(prediction)

    # 예측 결과 출력
    print("이미지에 있는 숫자는:", predicted_class)


    # # 숫자 예측
    # prediction = model.predict_classes(img_processed)

    # # 예측 결과 출력
    # print("이미지에 있는 숫자는:", prediction[0])

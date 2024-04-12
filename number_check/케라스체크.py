from keras.models import Sequential
from keras.layers import Dense

# 간단한 Sequential 모델 생성
model = Sequential()

# 입력 레이어와 히든 레이어 추가
model.add(Dense(units=64, activation='relu', input_dim=10))  # 입력 레이어
model.add(Dense(units=32, activation='relu'))  # 히든 레이어

# 출력 레이어 추가
model.add(Dense(units=1, activation='sigmoid'))  # 출력 레이어

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련 (더미 데이터 사용)
import numpy as np
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 모델 요약 정보 출력
model.summary()

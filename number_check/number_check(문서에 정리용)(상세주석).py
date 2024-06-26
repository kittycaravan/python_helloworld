import cv2              # 이미지 처리를 위한 OpenCV 라이브러리를 불러옵니다.
import numpy as np      # 숫자 연산을 위한 NumPy 라이브러리를 불러옵니다.
import tensorflow as tf # 머신 러닝 작업을 위한 TensorFlow 라이브러리를 불러옵니다.

#### [0/8].MNIST 데이터셋 로드
# TensorFlow의 내장 데이터셋에서 MNIST 데이터셋을 불러옵니다.
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#### [1/8].데이터 전처리
# 훈련 및 테스트 이미지의 픽셀 값을 0에서 1 사이로 정규화합니다.
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
**** 인공신경망 ****
인공신경망은 인간의 뇌를 모방한 컴퓨터 알고리즘입니다. 이를테면, 우리 뇌의 뉴런들이 정보를 받아들이고 처리하여 의사 결정을 내리는 과정을 모방한 것입니다.

이러한 네트워크는 뉴런이라는 단위로 구성되어 있습니다. 각 뉴런은 입력을 받아들이고, 이를 처리한 후 결과를 출력합니다. 
뉴런들은 서로 연결되어 있으며, 입력 신호는 각각의 연결에 가중치라는 값이 곱해져 전달됩니다. 
뉴런은 이러한 입력 신호와 가중치의 합을 활성화 함수를 사용하여 처리한 후, 결과를 출력합니다.

이러한 인공신경망은 다양한 형태와 구조를 가질 수 있습니다. 예를 들어, 입력층, 은닉층, 출력층 등으로 구성된 다층 신경망이 가장 일반적인 형태입니다. 
또한, 컨볼루션 신경망(CNN), 순환 신경망(RNN) 등의 다양한 유형이 존재합니다.

인공신경망은 주어진 입력 데이터를 학습하여 원하는 출력을 생성하도록 훈련됩니다. 이를 통해 패턴 인식, 분류, 예측 등 다양한 작업을 수행할 수 있습니다. 
인공신경망은 빅데이터와 함께 사용되어 이미지 인식, 자연어 처리, 음성 인식 등 다양한 영역에서 혁신적인 결과를 이끌어내고 있습니다.

**** Sequential 모델 ****
레이어를 순차적으로 쌓아 만든 인공 신경망.

- 다른 인공신경망 모델들

다층 퍼셉트론(MLP, Multi-Layer Perceptron): 
    여러 개의 은닉층을 갖는 피드포워드 신경망입니다. 각 층은 완전히 연결되어 있습니다.

컨볼루션 신경망(CNN, Convolutional Neural Network): 
    이미지 분류 및 컴퓨터 비전 작업에 주로 사용됩니다. 입력 이미지의 지역적 구조를 이해하고 학습하는 데 특화된 구조를 갖습니다.

순환 신경망(RNN, Recurrent Neural Network): 
    순차적인 데이터, 예를 들어 텍스트나 시계열 데이터를 처리하기에 적합한 구조를 갖습니다. 이전 단계의 출력이 다음 단계의 입력으로 사용됩니다.

장단기 메모리(LSTM, Long Short-Term Memory): 
    RNN의 한 종류로, 시퀀스 데이터의 장기 의존 관계를 모델링할 수 있는 더 긴 기간의 메모리를 유지합니다.

자기주도학습(SAE, Self-Organizing Map): 
    비지도 학습을 위한 인공 신경망 모델로, 데이터의 내재된 구조를 찾는 데 사용됩니다.

오토인코더(Autoencoder): 
    비지도 학습을 위한 신경망으로, 입력 데이터의 압축된 표현을 학습하여 잠재 변수를 학습하는 데 사용됩니다.

GAN(Generative Adversarial Network): 
    생성 모델로, 생성기와 판별기라는 두 개의 네트워크가 적대적 학습을 통해 데이터를 생성하고 평가합니다.

**** "Dense" 레이어 ****

"Dense" 레이어는 완전 연결층(Fully Connected Layer)을 나타냅니다. 
이 레이어는 이전 레이어의 모든 뉴런이 다음 레이어의 모든 뉴런과 연결되어 있는 것을 의미합니다.

Dense 레이어는 각 뉴런이 이전 레이어의 모든 입력과 연결되어 있기 때문에 "밀집층"이라고도 불립니다. 
이 연결은 가중치(weight)와 편향(bias)로 정의되며, 이러한 가중치와 편향은 신경망이 데이터에서 패턴을 학습하는 데 사용됩니다.

Dense 레이어는 입력과 출력을 모두 고려하여 모델의 복잡도를 높일 수 있습니다. 
이전 레이어의 모든 뉴런이 다음 레이어의 모든 뉴런에 연결되기 때문에 Dense 레이어는 데이터 간의 복잡한 관계를 모델링할 수 있습니다.

딥러닝에서는 Dense 레이어가 신경망의 핵심 구성 요소 중 하나입니다. 
일반적으로 입력 레이어 뒤에 나타나며, 중간 레이어 또는 출력 레이어로 사용될 수 있습니다.

- 하는일들

가중치 부여:

편향(Bias) 추가: 
    각 뉴런에는 가중치와 함께 편향(bias)이 추가됩니다. 
    편향은 해당 뉴런이 얼마나 쉽게 활성화되는지를 제어하는 값으로, 입력 신호와 가중치의 합에 더해집니다.

활성화 함수(Activation Function) 적용: 
    Dense 레이어의 각 뉴런은 활성화 함수를 사용하여 출력을 생성합니다. 
    활성화 함수는 뉴런의 출력을 결정하는 비선형 함수로, 네트워크가 복잡한 패턴을 학습할 수 있도록 돕습니다.

입력 데이터 변환: 
    입력 데이터가 Dense 레이어로 전달되기 전에는 일반적으로 형태를 변환합니다. 
    예를 들어, 2차원 이미지 데이터를 1차원 배열로 펼치는 작업이 있을 수 있습니다.

Regularization(정규화) 적용: 
    Dropout과 같은 정규화 기법이 Dense 레이어에 적용될 수 있습니다. 이러한 기법은 과적합을 방지하고 모델의 일반화 성능을 향상시키는 데 도움이 됩니다.

출력 생성: 
    Dense 레이어의 각 뉴런은 입력 신호와 가중치의 조합을 사용하여 출력을 생성합니다. 이러한 출력은 다음 레이어로 전달되거나 최종 출력으로 사용됩니다.

**** 케라스 ****
케라스(Keras)는 딥러닝 모델을 쉽게 구축하고 학습할 수 있도록 도와주는 고수준 딥러닝 라이브러리    

**** 과적합 ****
과적합(Overfitting)은 머신러닝 모델이 학습 데이터에 너무 맞춰져서 새로운 데이터에 대한 일반화 성능이 저하되는 현상을 말합니다.

'''
#### [2/8].모델 구성
# 레이어를 순차적으로 쌓아 모델을 구성합니다.
# 인공신경망 Sequential 모델을 사용
'''
드롭아웃 비율, ReLU 활성화 함수를 사용하는 은닉층의 뉴런 수, 그리고 출력층의 소프트맥스 뉴런 수와 같은 값들은 모델의 구조를 결정하는 매개변수들입니다. 
이러한 값들을 하이퍼파라미터(hyperparameter)라고 합니다.

최적의 드롭아웃 비율이나 뉴런 수는 주로 하이퍼파라미터 튜닝(hyperparameter tuning) 과정을 통해 결정됩니다. 
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 레이어 1 [Flatten] 28x28 이미지를 1차원 배열로 평평하게 펼치는 데 사용됩니다. 
                                                    #   입력 이미지의 형태를 변경하여 신경망의 첫 번째 은닉층에 전달합니다.
    tf.keras.layers.Dense(128, activation='relu'),  # 레이어 2 [Dense] 128개의 뉴런과 ReLU (Recitified Linear Unit) 활성화 함수를 갖는 Dense 레이어입니다.
                                                    #   이 은닉층은 입력층에서 받은 신호를 다음 층으로 전달하고, 
                                                    #   비선형성을 추가하여 신경망이 복잡한 패턴을 학습할 수 있도록 돕습니다.
    tf.keras.layers.Dropout(0.2),                   # 레이어 3 [Dropout] 0.2의 드롭아웃 비율을 적용합니다. 
                                                    #   드롭아웃은 훈련 중에 무작위로 일부 뉴런을 비활성화하여 과적합을 방지합니다. 
                                                    #   이를 통해 신경망이 더 일반적인 특징을 학습하도록 돕습니다.
                                                    #   ( 20% 가 드랍 됨 )
    tf.keras.layers.Dense(10, activation='softmax') # 레이어 4 [Dense] 10개의 뉴런과 소프트맥스 활성화 함수를 갖는 출력 레이어입니다. 
                                                    #   이 레이어는 신경망의 출력을 생성하며, 각 클래스에 대한 확률 분포를 반환합니다.
])

#### [3/8].모델 컴파일
# Adam 옵티마이저, 희소 카테고리컬 크로스엔트로피 손실 함수, 정확도 지표를 사용하여 모델을 컴파일합니다.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#### [4/8].모델 훈련
# 훈련 데이터로 모델을 5번 에포크 동안 훈련합니다.
model.fit(train_images, train_labels, epochs=5)

#### [5/8].예측을 위해 올려둔 이미지 파일 읽기
# '7.bmp'라는 이미지 파일을 읽어옵니다.
img = cv2.imread('7.bmp')   

#### [6/8].이미지 전처리
'''
**** 그레이스케일 변환 ****
컬러 이미지를 0(검은색)~255(흰색) 까지의 명암 수치를 포함한 단순화 처리하는 것
'''
# 이미지를 그레이스케일로 변환하고 이진 반전 임계값 처리를 적용합니다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

# 이미지를 28x28 픽셀 크기로 리사이징합니다.
resized = cv2.resize(thresh, (28, 28))

# 처리된 이미지를 NumPy 배열로 변환하고 픽셀 값을 0에서 1 사이로 정규화합니다.
img_processed = np.array(resized, dtype=np.float32).reshape(1, 28, 28) / 255.0

#### [7/8].숫자 예측
# 훈련된 모델을 사용하여 처리된 이미지에 대한 예측을 수행합니다.
prediction = model.predict(img_processed)
# 가장 높은 확률을 갖는 클래스를 예측 클래스로 선택합니다.
predicted_class = np.argmax(prediction)

#### [8/8].예측 결과 출력
# 이미지에 대한 예측된 클래스를 출력합니다.
print("이미지에 있는 숫자는:", predicted_class)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 긍정적인 문장들
positive_sentences = [
    "맛있어요",
    "맛집",
    "좋아요"
]

# 부정적인 문장들
negative_sentences = [
    "별로",
    "쓰레기"
]

# 각 문장에 대한 레이블 (0: 부정, 1: 긍정)
labels = [1, 1, 1, 0, 0]

# Tokenizer 객체 생성 및 문장들에 대해 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(positive_sentences + negative_sentences)

# 텍스트 시퀀스를 정수 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(positive_sentences + negative_sentences)

# 시퀀스를 일정한 길이로 패딩하여 같은 길이의 시퀀스를 만듦
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# 감정 분석 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(padded_sequences, labels, epochs=10)

# 새로운 문장에 대해 감정 분석 예측
test_sentence = "맛이 별로에요"
test_sequence = tokenizer.texts_to_sequences([test_sentence])
padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')
prediction = model.predict(padded_test_sequence)

if prediction >= 0.5:
    print("긍정적인 문장입니다.")
else:
    print("부정적인 문장입니다.")

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
positive_sentences = [
    "소식",
    "소식함",
    "샐러드 좋아함",
    "골고루 먹음",
    "싱거운",
    "싱거운거",
    "싱거운 음식",
    "싱겁게 먹음",
    "부드러운",
    "부드러운거",
    "부드러운 음식",
    "담백",
    "담백함",
    "담백한거",
    "담백한 음식",
    "담백하게 먹음",
    "고양이"
]
negative_sentences = [
    "대식함",
    "짜게 먹음",
    "맵게 먹음",
    "많이 먹음",
    "간식 많이 먹음",
    "야식 많이 먹음",
    "회식 자주함",
    "고기 좋아함",
    "편식함",
    "편식을 하는편임",
    "단거",
    "짠거",
    "매운거",
    "매운걸",
    "매운 음식을",
    "매운것을",
    "마라맛",
    "단짠"
]
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(positive_sentences + negative_sentences)
sequences = tokenizer.texts_to_sequences(positive_sentences + negative_sentences)
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
labels = np.array(labels)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10)
test_sentence = "고양이"
test_sequence = tokenizer.texts_to_sequences([test_sentence])
padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')
prediction = model.predict(padded_test_sequence)
print("prediction값: " + str(prediction))
if prediction >= 0.5:
    print("긍정적인 문장입니다.")
else:
    print("부정적인 문장입니다.")
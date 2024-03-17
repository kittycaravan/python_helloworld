import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense


sentences = ['맛있어요', '맛집', '좋아요', '별로', '쓰레기']
labels = [1, 1, 1, 0, 0]  # 긍정: 1, 부정: 0


tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')


model = Sequential([
    Embedding(100, 16, input_length=len(max(sequences, key=len))),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(padded, labels, epochs=30)


test_sentences = ["맛있어요", "별로", "좋아요", "쓰레기"]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post')

model.predict(test_padded)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

positive_sentences = [
    "good news",
    "got some good news",
    "like salads",
    "eat well-balanced",
    "bland",
    "blandness",
    "bland food",
    "eat blandly",
    "smooth",
    "smoothness",
    "smooth food",
    "simple",
    "simplicity",
    "simple meal",
    "simple food",
    "eat simply",
]
negative_sentences = [
    "overeating",
    "eat salty",
    "eat spicy",
    "eat a lot",
    "eat a lot of snacks",
    "eat a lot of late-night snacks",
    "frequent dining out",
    "like meat",
    "eat picky",
    "tend to be picky eater",
    "sweet and salty",
    "salty",
    "spicy",
    "spicy food",
    "spicy food",
    "spicy things",
    "spicy flavor",
    "sweet and salty",
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

test_sentence = "overeating"
test_sequence = tokenizer.texts_to_sequences([test_sentence])
padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')
prediction = model.predict(padded_test_sequence)
# print("prediction값: " + str(prediction[0][0]))
print("prediction값: " + str(prediction))
if prediction >= 0.5:
    print("긍정적인 문장입니다.")
else:
    print("부정적인 문장입니다.")

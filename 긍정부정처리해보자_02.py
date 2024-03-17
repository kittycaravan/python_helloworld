import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Positive sentences
positive_sentences = [
    "Delicious",
    "Great restaurant",
    "Love it"
]

# Negative sentences
negative_sentences = [
    "Awful",
    "Terrible"
]

# Labels (0: Negative, 1: Positive)
labels = [1, 1, 1, 0, 0]

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(positive_sentences + negative_sentences)

# Convert text sequences to integer sequences
sequences = tokenizer.texts_to_sequences(positive_sentences + negative_sentences)

# Padding sequences
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Convert labels to numpy array
labels = np.array(labels)

# Sentiment analysis model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10)

# Predict sentiment for a new sentence
test_sentence = "It tastes awful"
test_sequence = tokenizer.texts_to_sequences([test_sentence])
padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')
prediction = model.predict(padded_test_sequence)

if prediction >= 0.5:
    print("Positive sentence")
else:
    print("Negative sentence")

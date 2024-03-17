from tensorflow.keras.preprocessing.text import Tokenizer

# 테스트할 문장들
sentences = [
    "This is a test sentence.",
    "Another test sentence.",
    "Yet another sentence for testing."
]

# Tokenizer 객체 생성
tokenizer = Tokenizer()

# 문장들에 대해 토큰화 수행
tokenizer.fit_on_texts(sentences)

# 토큰화 결과 출력
print("단어 인덱스:", tokenizer.word_index)

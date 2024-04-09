#request 는 get방식으로 파라미터 받아오려고 추가함
from flask import Flask, jsonify, request

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)

# '/get_cat' 엔드포인트에 대한 GET 요청을 처리하는 함수를 정의합니다.
@app.route('/get_cat', methods=['GET'])
def get_cat():

    positive_sentences = [
    "cat",
    "dog",
    "lion",
    ]
    negative_sentences = [
        "rat",
        "snake",
        "bird",
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

    #test word
    test_sentence = request.args.get('test') #파라미터로 넘어온 값을 빼서 변수에 대입함.

    test_sequence = tokenizer.texts_to_sequences([test_sentence])
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_test_sequence)
    print("prediction값: " + str(prediction))
    result=""
    if prediction >= 0.5:
        result="긍정적인 문장입니다."
    else:
        result="부정적인 문장입니다."
    print(result)


    # 고양이 데이터를 담은 딕셔너리를 생성합니다.
    # cat_data = {"animal": "고양이"}
    cat_data = {"result": result}

    # 생성한 딕셔너리를 JSON 형식으로 변환하여 응답합니다.
    # Flask의 jsonify 함수를 사용하여 딕셔너리를 JSON 형식으로 변환합니다.
    return jsonify(cat_data)

#### 아래 코드는 pj 폴더 내에 한군데만 있어야함.
#### 그래서 주석쳐둠. (날씨쪽에서 쓰고있음)
# 이 코드를 직접 실행할 때만 Flask 서버를 실행합니다.
# 다른 파일에서 이 코드를 import할 때는 실행되지 않습니다.
# if __name__ == '__main__':
#     # Flask 애플리케이션을 디버그 모드로 실행합니다.
#     # 디버그 모드를 사용하면 코드를 수정하고 저장할 때마다 서버가 자동으로 재시작됩니다.
    # app.run(debug=True)
    
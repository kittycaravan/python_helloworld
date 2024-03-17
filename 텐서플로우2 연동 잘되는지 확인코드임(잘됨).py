import tensorflow as tf

# TensorFlow 그래프 생성
@tf.function
def add_numbers(a, b):
    return a + b

# 그래프 실행
result = add_numbers(5, 3)
print("두 수의 합:", result.numpy())

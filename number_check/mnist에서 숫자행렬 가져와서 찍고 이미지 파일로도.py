# 필요한 라이브러리 불러오기
import numpy
import matplotlib.pyplot
import csv
import urllib.request
import codecs

# mnist 학습 데이터인 웹에 있는 csv 파일을 리스트로 불러오기
url = 'https://media.githubusercontent.com/media/freebz/Make-Your-Own-Neural-Network/master/mnist_dataset/mnist_test_10.csv'

# 웹 데이터 열기
response = urllib.request.urlopen(url)

# csv 파일 읽기
training_data_file = csv.reader(codecs.iterdecode(response, 'utf-8'))
training_data_list = list(training_data_file)  # 리스트로 만들기
all_values = training_data_list[0]  # 데이터 한 개

# 저장된 숫자가 무엇인지 파악하기
print(all_values[0])    # 답은 7

# 28 x 28 행렬로 바꾸기
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))

# 행렬을 이미지로 시각화하기
matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None')

# 이미지를 foo.png로 저장
matplotlib.pyplot.savefig('foo.png')

# 웹 데이터 닫기
response.close()
#request 는 get방식으로 파라미터 받아오려고 추가함
from flask import Flask, jsonify, request

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)

# '/get_cat' 엔드포인트에 대한 GET 요청을 처리하는 함수를 정의합니다.
@app.route('/get_cat', methods=['GET'])
def get_cat():

    name = request.args.get('name') #name파라미터로 넘어온 값을 빼서 name 변수에 대입함.

    # 고양이 데이터를 담은 딕셔너리를 생성합니다.
    # cat_data = {"animal": "고양이"}
    cat_data = {"animal": name}
    # 생성한 딕셔너리를 JSON 형식으로 변환하여 응답합니다.
    # Flask의 jsonify 함수를 사용하여 딕셔너리를 JSON 형식으로 변환합니다.
    return jsonify(cat_data)

# 이 코드를 직접 실행할 때만 Flask 서버를 실행합니다.
# 다른 파일에서 이 코드를 import할 때는 실행되지 않습니다.
if __name__ == '__main__':
    # Flask 애플리케이션을 디버그 모드로 실행합니다.
    # 디버그 모드를 사용하면 코드를 수정하고 저장할 때마다 서버가 자동으로 재시작됩니다.
    app.run(debug=True)

from flask import Flask, request, jsonify
from recommendation_mvp import recommend

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)

# '/recommend' 주소로 POST 요청을 처리할 API 엔드포인트를 정의합니다.
@app.route('/recommend', methods=['POST'])
def handle_recommendation():
    """
    클라이언트로부터 survey와 user_context를 JSON으로 받아
    추천 결과를 JSON으로 반환합니다.
    """
    # 1. 요청에서 JSON 데이터 추출
    data = request.get_json()
    if not data:
        return jsonify({"error": "잘못된 JSON 형식입니다."}), 400

    survey = data.get('survey')
    user_ctx = data.get('user_context')

    if not survey or not user_ctx:
        return jsonify({"error": "'survey' 또는 'user_context' 데이터가 없습니다."}), 400

    # 2. 핵심 추천 로직 호출
    try:
        # recommendation_mvp.py의 recommend 함수를 사용합니다.
        recommendations_df = recommend(survey=survey, user_ctx=user_ctx)

        # 3. 결과를 JSON으로 변환하여 반환
        # DataFrame을 클라이언트가 사용하기 쉬운 리스트-딕셔너리 형태로 변환합니다.
        result = recommendations_df.to_dict(orient='records')
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"입력된 survey 데이터에 '{e}' 키가 없습니다."}), 400
    except Exception as e:
        # 서버 로그에 에러를 기록하고, 클라이언트에게는 일반적인 에러 메시지를 보냅니다.
        print(f"서버 내부 오류 발생: {e}")
        return jsonify({"error": "추천을 생성하는 중 오류가 발생했습니다."}), 500

# 이 파일을 직접 실행했을 때 웹 서버를 시작합니다.
if __name__ == '__main__':
    # debug=True 모드로 실행하여 코드 변경 시 서버가 자동 재시작되도록 합니다.
    app.run(debug=True, port=5000)
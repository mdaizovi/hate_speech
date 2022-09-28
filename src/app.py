from flask import Flask, request,  jsonify

import random

app = Flask(__name__)


@app.route('/check-text', methods=['POST'])
def check_text_for_hate():
    request_json = request.get_json()
    td_list = [{"id": k, "text": v} for k, v in request_json.items()]

    # TODO replace with real score
    for d in td_list:
        d["score"] = random.randint(0, 10)

    print(td_list)
    return jsonify(td_list)


# run with 'python app.py'
if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""
curl -X POST -H "Content-Type: application/json" -d '{"0_1234": "test text am i toxic?", "1_876543": "more text"}' http://localhost:5000/check-text
"""

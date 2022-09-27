from flask import Flask, request,  jsonify

import random

app = Flask(__name__)


@app.route('/check-text', methods=['POST'])
def check_text_for_hate():
    # NOTE will need to get get bulk text in one API call, and recall the id so can affect parent's css.
    request_json = request.get_json()
    print(f"request_json {request_json}")
    score = random.randint(0, 100)
    return jsonify({"score": score})


# run with 'python app.py'
if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""
curl -X POST -H "Content-Type: application/json" -d '{"1": "test text am i toxic?"}' http://localhost:5000/check-text
"""

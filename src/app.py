from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap, Bootstrap4
from flask_cors import CORS

import random

app = Flask(__name__)
bootstrap = Bootstrap4(app)
CORS(app)


@app.route('/check-text', methods=['POST'])
def check_text_for_hate():
    request_json = request.get_json(force=True)
    # print(request_json)
    td_list = [{"id": k, "text": v} for k, v in request_json.items()]
    # print(td_list)
    # TODO replace with real score
    for d in td_list:
        d["score"] = random.randint(0, 10)

    print(td_list)
    return jsonify(td_list)


@app.route('/')
def index():
    return render_template('index.html', next="/definition")


@app.route('/definition')
def definiiton_view():
    return render_template('definition.html', next="/dataset")


@app.route('/dataset')
def dataset_view():
    return render_template('dataset.html', next="/models")


@app.route('/models')
def models_view():
    return render_template('models.html', next="/extension")


@app.route('/extension')
def extension_view():
    return render_template('extension.html', next="/future")


@app.route('/future')
def future_view():
    return render_template('future.html', next=None)


# run with 'python app.py'
if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""
curl -X POST -H "Content-Type: application/json" -d '{"0_1234": "test text am i toxic?", "1_876543": "more text"}' http://localhost:5000/check-text
"""

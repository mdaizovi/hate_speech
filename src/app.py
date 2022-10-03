from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap, Bootstrap4
from flask_cors import CORS

import random

from analyzer_kaggle_binary import*
ka = KaggleBinaryAnalyzer(train_filepath="kaggle_binary/train.csv",
                          test_filepath="kaggle_binary/test.csv")
ka.tt_split()
ka.vectorize()
ka.run_model(model_class=LogisticRegression)
print("\n\nmodel is ready")

app = Flask(__name__)
bootstrap = Bootstrap4(app)
CORS(app)


@app.route('/check-text', methods=['POST'])
def check_text_for_hate():
    # TODO
    # this works on my basic html but when i tried to test WAP in the wild I'm not getting a lot of the actual text
    # examples real world sites to try again:
    # https://www.songtexte.com/songtext/cardi-b/wap-g63b72a0f.html
    # https://www.azlyrics.com/lyrics/cardi-b/wap.html
    # Guess I need to re-do front end to send BE everything,
    request_json = request.get_json(force=True)
    print(f"\n\nrequest_json {request_json}\n\n")
    columns = ["id", "comment_text"]
    data = [[k, v] for k, v in request_json.items()]
    df = pd.DataFrame(data=data, columns=columns)
    score_df = ka.predict_unlabeled_df(df)
    return_d = score_df["score"].to_dict()
    # Should I only bkur text that's .5 or greater, or let questionable text be slightly blurred?
    #td_list = [{"id":k,"score": (5 * v if v >= 0.5 else v)} for k, v in return_d.items()]
    #td_list = [{"id":k,"score": (5 * v)} for k, v in return_d.items()]
    td_list = [{"id":k,"score": v} for k, v in return_d.items()]
    for i in td_list:
        print(i)
    return jsonify(td_list)


@app.route('/')
def index():
    return render_template('index.html', next="/definition")


@app.route('/definition')
def definition_view():
    return render_template('definition.html', next="/dataset")


@app.route('/dataset')
def dataset_view():
    return render_template('dataset.html', next="/prep")


@app.route('/prep')
def prep_view():
    return render_template('prep.html', next="/models")


@app.route('/models')
def models_view():
    return render_template('models.html', next="/extension")


@app.route('/extension')
def extension_view():
    return render_template('extension.html', next="/limits")

@app.route('/limits')
def limit_view():
    return render_template('limitations.html', next="/future")

@app.route('/future')
def future_view():
    return render_template('future.html', next=None)


# run with 'python app.py'
if __name__ == "__main__":
    app.run(debug=True, port=5000)
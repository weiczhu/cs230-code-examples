from flask import Flask
from flask import Response

import json
from predict import get_predict_params
from predict import Predictor

from trie import get_dictionary_path
from trie import Trie

def create_app(debug=False):
    app = Flask(__name__)
    app.debug = debug

    return app

app = create_app(debug=True)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/location/extract/<sentence>')
def location_extract(sentence):

    sentences_texts = [sentence]

    predictions = predictor.predict(sentences_texts)

    trie.pos_tag(sentence)

    json_result = json.dumps({'data': predictions[0], 'success': True})
    resp = Response(json_result, status=200, mimetype='application/json')

    return resp


@app.route('/location/suggestion/<sentence>')
def location_suggestion(sentence):

    suggestions = trie.pred_suggest(sentence)

    json_result = json.dumps({'data': suggestions, 'success': True})
    resp = Response(json_result, status=200, mimetype='application/json')

    return resp


if __name__ == "__main__":

    predictor = Predictor(get_predict_params())
    trie = Trie(get_dictionary_path())

    app.run(port=8081)
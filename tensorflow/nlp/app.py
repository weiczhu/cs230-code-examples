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

    pos_tags = trie.pos_tag(sentence)
    pos_tags.sort(key=lambda x: len(x[0]), reverse=True)
    print('pos_tags:', pos_tags)

    existing_words = list()
    filtered_pos_tags = list()
    for pos_tag in pos_tags:
        word = pos_tag[0]
        tags = pos_tag[1]

        existed = False
        for existing_word in existing_words:
            if word in existing_word:
                existed = True
                break
        if existed:
            continue

        filtered_pos_tags.append([word, tags])
        existing_words.append(word)

    pos_tags = filtered_pos_tags

    # Hybrid approach for NER
    concept_objects = list()
    hypothesis_pos_tags = list()
    for pos_tag in pos_tags:
        word = pos_tag[0]
        tags = pos_tag[1]

        # Decide if it's a determined word (is in the dictionary and not ambiguous), confidence 1.0
        if len(tags) == 1:
            concept_object = {'value': word, 'class': tags[0], 'confidence': 1.0}
            concept_objects.append(concept_object)
        elif len(tags) > 1:
            hypothesis_pos_tags.append(pos_tag)

    for prediction in predictions[0]:
        pred_word = prediction[0].lower()
        pred_tag = prediction[1]

        # Decide if it's a discovered word (not in the dictionary), confidence 0.7
        is_discover_word = True
        for concept_object in concept_objects:
            if pred_word == concept_object['value']:  # and pred_tag == concept_object['class']:
                is_discover_word = False

        if is_discover_word:
            concept_object = {'value': pred_word, 'class': pred_tag, 'confidence': 0.7}
            concept_objects.append(concept_object)

        # Decide if it's an ambiguous word (in the dictionary but with multiple tags), confidence 0.9
        for hypothesis_pos_tag in hypothesis_pos_tags:
            word = hypothesis_pos_tag[0]
            tags = hypothesis_pos_tag[1]

            if pred_word == word and pred_tag in tags:
                concept_object = {'value': pred_word, 'class': pred_tag, 'confidence': 0.9}
                concept_objects.append(concept_object)

    json_result = json.dumps({'data': concept_objects, 'success': True})
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

"""Tensorflow utility functions for evaluation"""

from model.model_fn import viterbi_prediction


def preprocess_input(sentences, params):

    path_words = params.path_words
    path_tags = params.path_tags

    word2idx = dict()
    idx2word = dict()
    tag2idx = dict()
    idx2tag = dict()

    with open(path_words) as f:
        for idx, line in enumerate(f):
            word = line.strip()
            if word:
                word2idx[word] = idx
                idx2word[idx] = word

    with open(path_tags) as f:
        for idx, line in enumerate(f):
            tag = line.strip()
            if tag:
                tag2idx[tag] = idx
                idx2tag[idx] = tag

    params.word2idx = word2idx
    params.idx2word = idx2word
    params.tag2idx = tag2idx
    params.idx2tag = idx2tag

    # Load Vocabularies
    id_pad_word = word2idx[params.pad_word]
    id_pad_tag = tag2idx[params.pad_tag]
    print('id_pad_word:', id_pad_word)
    print('id_pad_tag:', id_pad_tag)

    default_idx_for_unk = len(word2idx)
    sentences = [list(map(lambda x: word2idx.get(x, default_idx_for_unk), sentence.lower().split(' '))) for sentence in
                 sentences]
    sentence_lengths = list(map(lambda x: len(x), sentences))
    max_len = max(sentence_lengths)
    sentences = [a + [id_pad_word] * (max_len - len(a)) for a in sentences]

    return sentences, sentence_lengths


def decode_logits(logits, sentences_texts, sentence_lengths, trans_params_value, idx2tag):

    predictions = viterbi_prediction(list(logits), sentence_lengths, trans_params_value)

    predictions = [list(map(lambda x: idx2tag.get(x), prediction)) for prediction in
                   predictions]
    print('predictions:', predictions)

    recognized_entities_list = list()
    for i in range(len(sentences_texts)):
        sentence = sentences_texts[i].split(' ')
        prediction = predictions[i]

        recognized_entities = list()

        for (word, tag) in zip(sentence, prediction):

            if 'B-' in tag:
                recognized_entities.append([word, tag])
            elif 'I-' in tag and len(recognized_entities) > 0:
                prev_word, prev_tag = recognized_entities[-1]

                if 'B-' in prev_tag or 'I-' in prev_tag and (prev_tag[2:] == tag[2:]):
                    word = prev_word + ' ' + word
                    recognized_entities[-1] = [word, tag]

        recognized_entities = list(map(lambda x: [x[0], x[1][2:]], recognized_entities))

        recognized_entities_list.append(recognized_entities)

    return recognized_entities_list

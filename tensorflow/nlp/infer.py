"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.inference import infer
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_slices
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'infer.log'))

    # Get paths for vocabularies and dataset
    path_words = os.path.join(args.data_dir, 'words.txt')
    path_tags = os.path.join(args.data_dir, 'tags.txt')
    path_eval_sentences = os.path.join(args.data_dir, 'test/sentences.txt')
    path_eval_labels = os.path.join(args.data_dir, 'test/labels.txt')

    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=num_oov_buckets)
    tags = tf.contrib.lookup.index_table_from_file(path_tags)

    words_idx2str = tf.contrib.lookup.index_to_string_table_from_file(path_words)
    tags_idx2str = tf.contrib.lookup.index_to_string_table_from_file(path_tags)

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    sentences = ['Miyasaka is a good place',
                 'I am working at tsurumaki',
                 'I want to go to tamagwa setagaya tokyo',
                 'I live at tamagawa denenchofu setagaya',
                 'I favor Gotokuji setagaya-ku for our trip',
                 'shall we go to tamagawa denenchofu setagaya-ku tokyo 158-0085 for lunch ?',
                 'kasuya setagaya-ku tokyo 158-0085 is my address',
                 'deliver it to kasuya setagaya-ku tokyo 158-0085']

    sentences = [x.lower() for x in sentences]

    infer_sentences = load_dataset_from_slices(sentences, words)

    # Specify other parameters for the dataset and the model
    params.batch_size = len(sentences)
    params.id_pad_word = words.lookup(tf.constant(params.pad_word))
    params.id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create iterator over the test set
    inputs = input_fn('infer', infer_sentences, None, params)
    inputs['tags_idx2str'] = tags_idx2str

    if params.model_version == 'lstm-crf':
        best_trans_params_path = os.path.join(args.model_dir, "metrics_eval_best_trans_params.npy")
        trans_params = np.load(best_trans_params_path)
        inputs['trans_params'] = trans_params

    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('infer', inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting inference")
    metrics = infer(model_spec, args.model_dir, params, args.restore_from)
    predictions = metrics['predictions']

    print('---------------------------------------')
    print('model inference results:')
    annotated_sents = []
    for (sentence, prediction) in zip(sentences, predictions):
        sentence = sentence.split(' ')
        prediction = prediction.split(' ')
        annotated_sent = []
        for (s, p) in zip(sentence, prediction):
            annotated_sent.append(s + '/' + p)
        annotated_sent = ' '.join(annotated_sent)

        annotated_sents.append(annotated_sent)
        print(annotated_sent)

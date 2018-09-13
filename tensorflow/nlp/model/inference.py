"""Tensorflow utility functions for evaluation"""

import logging
import os

import numpy as np
from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json
from model.model_fn import viterbi_prediction

def inference_sess(sess, model_spec, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])

    if params.model_version == 'lstm':
        predictions, sentence_lengths = sess.run([model_spec['predictions'],
                            model_spec['sentence_lengths']])

        idx2tags = model_spec['idx2tags']
        metrics_val = dict()
        pred_sentences = []
        for i, prediction in enumerate(predictions):
            sentence_length = sentence_lengths[i]
            pred_sentence = []
            for j, word_id in enumerate(prediction):
                if j < sentence_length:
                    pred_sentence.append(idx2tags.lookup(tf.constant(word_id, dtype='int64')))
            pred_sentences.append(pred_sentence)

        pred_sentences = sess.run(pred_sentences)
        pred_sentences = [' '.join([word.decode('utf-8') for word in pred_sentence]) for pred_sentence in pred_sentences]
        metrics_val['predictions'] = pred_sentences

    elif params.model_version == 'lstm-crf':
        # get tag scores and transition params of CRF
        logits, sentence_lengths = sess.run([model_spec['logits'],
                                            model_spec['sentence_lengths']])
        trans_params = model_spec['trans_params']

        predictions = viterbi_prediction(logits, sentence_lengths, trans_params)

        idx2tags = model_spec['idx2tags']
        metrics_val = dict()
        pred_sentences = []
        for i, prediction in enumerate(predictions):
            sentence_length = sentence_lengths[i]
            pred_sentence = []
            for j, word_id in enumerate(prediction):
                if j < sentence_length:
                    pred_sentence.append(idx2tags.lookup(tf.constant(word_id, dtype='int64')))
            pred_sentences.append(pred_sentence)

        pred_sentences = sess.run(pred_sentences)
        pred_sentences = [' '.join([word.decode('utf-8') for word in pred_sentence]) for pred_sentence in
                          pred_sentences]
        metrics_val['predictions'] = pred_sentences

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    print('metrics_val:', metrics_val)
    return metrics_val


def infer(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)

        print('save_path:\n', save_path)
        saver.restore(sess, save_path)

        # Evaluate
        metrics = inference_sess(sess, model_spec, params=params)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_infer_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)

        return metrics

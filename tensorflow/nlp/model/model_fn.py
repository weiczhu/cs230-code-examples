"""Define the model."""

import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    trans_params = tf.get_variable(name="trans_params", dtype=tf.float32,
                                   shape=[params.number_of_tags, params.number_of_tags])

    sentences = features['sentences']
    sentence_lengths = features['sentence_lengths']

    sentence_lengths = tf.cast(sentence_lengths, tf.int32)

    # Get word embeddings for each token in the sentence
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[params.vocab_size, params.embedding_size])
    embedding_lookup = tf.nn.embedding_lookup(embeddings, sentences)

    # Apply LSTM over the embeddings
    cell_fw = tf.contrib.rnn.LSTMCell(params.lstm_num_units)
    cell_bw = tf.contrib.rnn.LSTMCell(params.lstm_num_units)

    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                cell_bw, embedding_lookup,
                                                                sequence_length=sentence_lengths,
                                                                time_major=False,
                                                                dtype=tf.float32)

    output = tf.concat([output_fw, output_bw], axis=-1)

    # Compute logits from the output of the LSTM
    logits = tf.layers.dense(output, params.number_of_tags)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.

        predictions, _ = tf.contrib.crf.crf_decode(logits, trans_params, sentence_lengths)

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs={
                                              'predictions': tf.estimator.export.PredictOutput(predictions)
                                          },
                                          prediction_hooks=[]
                                          )
    else:

        # trans_params_update_o = tf.assign(trans_params, transition_params)
        # Define loss and accuracy (we need to apply a mask to account for padding)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sentence_lengths)
        loss = tf.reduce_mean(-log_likelihood)

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.assign(trans_params, transition_params)

        predictions, _ = tf.contrib.crf.crf_decode(logits, trans_params, sentence_lengths)

        # Define training step that minimizes the loss with the Adam optimizer
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)

        mask = tf.sequence_mask(sentence_lengths)
        # apply mask
        labels_masked = tf.boolean_mask(labels, mask)
        predictions_masked = tf.boolean_mask(predictions, mask)

        with tf.variable_scope("metrics"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                metrics = {
                    'loss': tf.metrics.mean(loss),
                    'accuracy': tf.metrics.accuracy(labels_masked, predictions_masked)
                }
            elif mode == tf.estimator.ModeKeys.EVAL:
                metrics = {
                    'eval_loss': tf.metrics.mean(loss),
                    'eval_accuracy': tf.metrics.accuracy(labels_masked, predictions_masked)
                }

        feed_dict = {
                    'metrics': metrics,
                    'global_step': global_step
        }
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            training_hooks=[TrainEvalHook(mode, feed_dict, params)],
            evaluation_hooks=[TrainEvalHook(mode, feed_dict, params)]
            )

    return spec


def viterbi_prediction(logits, sentence_lengths, trans_params):

    viterbi_sequences = []
    viterbi_scores = []

    # iterate over the sentences because no batching in vitervi_decode
    for logit, sequence_length in zip(logits, sentence_lengths):
        logit = logit[:sequence_length]  # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
            logit, trans_params)
        viterbi_sequences += [viterbi_seq]
        viterbi_scores += [viterbi_score]

    predictions = viterbi_sequences, viterbi_scores

    return predictions, viterbi_scores


def compute_accuracy(predictions, sentence_lengths, labels):
    accuracy = list()

    for lab, lab_pred, length in zip(labels, predictions, sentence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]
        accuracy += [a == b for (a, b) in zip(lab, lab_pred)]

    return np.mean(accuracy)


class TrainEvalHook(tf.train.SessionRunHook):

    def __init__(self, mode, feed_dict, params):
        self.loss_hist = list()
        self.accuracy_hist = list()

        self.mode = mode
        self.feed_dict = feed_dict
        self.params = params

        self.feed_dict_value = None

    def begin(self):
        pass

    def end(self, session):
        if self.mode == tf.estimator.ModeKeys.EVAL:
            global_step = self.feed_dict_value['global_step']
            epochs = global_step // self.params.num_steps

            print('Epochs {:3}:\t{}\tlosses:{:05.3f};\taccuracy:{:05.3f}'.format(epochs, self.mode,
                                                                                 np.mean(self.loss_hist),
                                                                                 np.mean(self.accuracy_hist)))

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'feed_dict': self.feed_dict})

    def after_run(self, run_context, run_values):
        self.feed_dict_value = run_values.results['feed_dict']
        metrics = self.feed_dict_value['metrics']
        global_step = self.feed_dict_value['global_step']

        epochs = global_step // self.params.num_steps
        steps = global_step % self.params.num_steps

        # print('metrics:', metrics)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.loss_hist.append(metrics['loss'][1])
            self.accuracy_hist.append(metrics['accuracy'][1])
        else:
            self.loss_hist.append(metrics['eval_loss'][1])
            self.accuracy_hist.append(metrics['eval_accuracy'][1])

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            print('Epochs {:3}\tsteps {:3}:\t{}\tlosses:{:05.3f};\taccuracy:{:05.3f}'.format(epochs, steps, self.mode,
                                                                             self.loss_hist[-1],
                                                                             self.accuracy_hist[-1]))


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

        logits, _ = tf.contrib.crf.crf_decode(logits, trans_params, sentence_lengths)

        feed_dict = {
            'logits': logits,
            'sentence_lengths': sentence_lengths,
            # 'trans_params': trans_params
        }

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=logits,
                                          export_outputs={
                                              'predictions': tf.estimator.export.PredictOutput(logits)
                                          },
                                          prediction_hooks=[PredictHook(feed_dict, params)]
                                          )
    else:

        # trans_params_update_o = tf.assign(trans_params, transition_params)
        # Define loss and accuracy (we need to apply a mask to account for padding)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sentence_lengths)
        loss = tf.reduce_mean(-log_likelihood)

        tf.assign(trans_params, transition_params)

        logits, _ = tf.contrib.crf.crf_decode(logits, trans_params, sentence_lengths)

        # Define training step that minimizes the loss with the Adam optimizer
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("metrics"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                metrics = {
                    'loss': tf.metrics.mean(loss)
                }
            elif mode == tf.estimator.ModeKeys.EVAL:
                metrics = {
                    'eval_loss': tf.metrics.mean(loss)
                }

        feed_dict = {
                    'loss': loss,
                    'logits': logits,
                    'labels': labels,
                    'sentence_lengths': sentence_lengths,
                    # 'trans_params_update_o': trans_params_update_o,
                    # 'trans_params': trans_params,
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

    return predictions


def compute_accuracy(logits, sentence_lengths, trans_params, labels):
    accuracy = list()
    # predictions, _ = viterbi_prediction(logits, sentence_lengths, trans_params)
    predictions = logits

    for lab, lab_pred, length in zip(labels, predictions, sentence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]
        accuracy += [a == b for (a, b) in zip(lab, lab_pred)]

    return np.mean(accuracy)


class PredictHook(tf.train.SessionRunHook):

    def __init__(self, feed_dict, params):
        self.predictions_hist = list()

        self.feed_dict = feed_dict
        self.params = params

    def begin(self):
        pass

    def end(self, session):
        # print('predictions:', self.predictions_hist)
        pass

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'feed_dict': self.feed_dict})

    def after_run(self, run_context, run_values):
        self.feed_dict_value = run_values.results['feed_dict']

        logits = self.feed_dict_value['logits']
        sentence_lengths = self.feed_dict_value['sentence_lengths']
        # trans_params = self.feed_dict_value['trans_params']
        # print('trans_params:', trans_params)

        # predictions, _ = viterbi_prediction(logits, sentence_lengths, trans_params)
        predictions = logits
        self.predictions_hist.append(predictions)


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
            self.epoch_end()

    def epoch_end(self):
        mode_str = 'Training   ' if self.mode == tf.estimator.ModeKeys.TRAIN else 'Evaluation'
        global_step = self.feed_dict_value['global_step']
        epochs = global_step // self.params.num_steps

        loss_hist = self.loss_hist[-self.params.num_steps:] if self.mode == tf.estimator.ModeKeys.TRAIN \
                                                            else self.loss_hist
        accuracy_hist = self.accuracy_hist[-self.params.num_steps:] if self.mode == tf.estimator.ModeKeys.TRAIN \
            else self.accuracy_hist

        print('Epochs\t{}: {}\t\tlosses:{:05.3f};\taccuracy:{:05.3f}'.format(epochs, mode_str,
                                                                             np.mean(loss_hist),
                                                                             np.mean(accuracy_hist)))

        # print('saved trans_params:', self.feed_dict_value['trans_params'])
        # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #     np.save(self.params.trans_params_path, self.feed_dict_value['trans_params'])

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'feed_dict': self.feed_dict})

    def after_run(self, run_context, run_values):
        self.feed_dict_value = run_values.results['feed_dict']

        loss = self.feed_dict_value['loss']
        logits = self.feed_dict_value['logits']
        # trans_params = self.feed_dict_value['trans_params']
        labels = self.feed_dict_value['labels']
        sentence_lengths = self.feed_dict_value['sentence_lengths']
        global_step = self.feed_dict_value['global_step']

        self.loss_hist.append(loss)
        self.accuracy_hist.append(compute_accuracy(logits, sentence_lengths, None, labels))

        if self.mode == tf.estimator.ModeKeys.TRAIN and global_step % self.params.num_steps == 0:
            self.epoch_end()


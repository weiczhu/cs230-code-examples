"""Train the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.process_fn import preprocess_input
from model.process_fn import decode_logits


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")



# Set the random seed for the whole graph for reproductible experiments
tf.set_random_seed(230)

# Load the parameters from the experiment params.json file in model_dir
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

# Load the parameters from the dataset, that gives the size etc. into params
json_path = os.path.join(args.data_dir, 'dataset_params.json')
assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
params.update(json_path)
num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

# Check that we are not overwriting some previous experiment
# Comment these lines if you are developing your model and don't care about overwritting
model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "checkpoint"))
overwritting = model_dir_has_best_weights and args.restore_dir is None
# assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

# Set the logger
set_logger(os.path.join(args.model_dir, 'train.log'))


# Specify other parameters for the dataset and the model
params.eval_size = params.dev_size
params.buffer_size = params.train_size # buffer size for shuffling

params.path_words = os.path.join(args.data_dir, 'words.txt')
params.path_tags = os.path.join(args.data_dir, 'tags.txt')

def train_input_fn():
    path_train_sentences = os.path.join(args.data_dir, 'train/sentences.txt')
    path_train_labels = os.path.join(args.data_dir, 'train/labels.txt')

    return input_fn('train', path_train_sentences, path_train_labels, params)


def eval_input_fn():
    path_test_sentences = os.path.join(args.data_dir, 'dev/sentences.txt')
    path_test_labels = os.path.join(args.data_dir, 'dev/labels.txt')

    return input_fn('eval', path_test_sentences, path_test_labels, params)


def serving_input_receiver_fn():

    receiver_tensors = {
        'sentences': tf.placeholder(tf.int64, [None, None]),
        'sentence_lengths': tf.placeholder(tf.int64, [None])
    }

    # Convert give inputs to adjust to the model.
    features = receiver_tensors
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
                                                    features=features)


if __name__ == '__main__':

    logging.info("Start training model")

    trans_params_path = os.path.join(args.model_dir, "trans_params.npy")
    params.trans_params_path = trans_params_path

    num_steps = params.train_size // params.batch_size
    params.num_steps = num_steps

    chkpt_path = os.path.join(args.model_dir, 'checkpoint')
    expt_path = 'exporter'

    run_config = tf.estimator.RunConfig(save_summary_steps=params.num_steps,
                                        save_checkpoints_steps=params.num_steps * params.num_epochs,
                                        log_step_count_steps=params.num_steps * params.num_epochs,
                                        keep_checkpoint_max=5)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params=params,
                                       model_dir=chkpt_path)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.num_epochs * params.num_steps)

    exporter = tf.estimator.LatestExporter(expt_path, serving_input_receiver_fn, exports_to_keep=5)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,  # no need to batch in eval
        steps=None,
        exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

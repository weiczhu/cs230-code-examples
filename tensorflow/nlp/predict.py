"""Train the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.model_fn import model_fn
from model.process_fn import preprocess_input
from model.process_fn import postprocess_output


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/crf_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/custom', help="Directory containing the dataset")
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

# Set the logger
set_logger(os.path.join(args.model_dir, 'train.log'))


# Specify other parameters for the dataset and the model
params.eval_size = params.dev_size
params.buffer_size = params.train_size # buffer size for shuffling

params.path_words = os.path.join(args.data_dir, 'words.txt')
params.path_tags = os.path.join(args.data_dir, 'tags.txt')

logging.info("Start training model")
trans_params_path = os.path.join(args.model_dir, "trans_params.npy")
params.trans_params_path = trans_params_path
checkpoint_path = os.path.join(args.model_dir, 'checkpoint')
params.checkpoint_path = checkpoint_path


# Expose params to outer scope
def get_predict_params():
    global params
    return params


# Predictor class for prediction serving
class Predictor:

    # Static variable for estimator
    estimator = None

    def __init__(self, params):
        self.params = params

        # Using lazy instantiation for estimator
        if Predictor.estimator is None:
            Predictor.estimator = tf.estimator.Estimator(model_fn=model_fn,
                                               params=self.params,
                                               model_dir=self.params.checkpoint_path)

        self.sentences_texts = None
        self.sentences, self.sentence_lengths = None, None

    def predict_input_fn(self):

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sentences': np.array(self.sentences), 'sentence_lengths': np.array(self.sentence_lengths)},
            batch_size=1,
            num_epochs=1,
            shuffle=False)

        return predict_input_fn()

    def predict(self, sentences_texts):

        self.sentences_texts = sentences_texts
        self.sentences, self.sentence_lengths = preprocess_input(self.sentences_texts, self.params)

        predictions = Predictor.estimator.predict(input_fn=self.predict_input_fn)

        recognized_entities_list = postprocess_output(predictions, sentences_texts, params)

        return recognized_entities_list


if __name__ == '__main__':

    predictor = Predictor(params)

    sentences_texts = ['I want to go to tamagwa setagaya tokyo',
                       'I live at tamagawa denenchofu setagaya',
                       'I favor Gotokuji setagaya-ku for our trip',
                       'shall we go to tamagawa denenchofu setagaya-ku tokyo 158-0085 for lunch ?',
                       'kasuya setagaya-ku tokyo 158-0085 is my address',
                       'deliver it to kasuya setagaya-ku tokyo 158-0085']

    print(predictor.predict(sentences_texts))

    sentences_texts = ['Miyasaka is a good place',
                       'I am working at tsurumaki']

    print(predictor.predict(sentences_texts))


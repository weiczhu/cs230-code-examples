"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch

import model.net as net
import utils
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def predict(model, data):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop

    # fetch the next evaluation batch
    data_batch = torch.LongTensor(data)

    # compute model output
    output_batch = model(data_batch)

    # extract data from torch Variable, move to cpu, convert to numpy arrays
    data_batch_aslist = data_batch.data.cpu().numpy().tolist()
    data_batch = [data_loader.tokenizer.convert_ids_to_tokens(x) for x in data_batch_aslist]

    output_batch = output_batch.data.cpu().numpy()
    confidence_batch = np.max(output_batch, axis=-1).tolist()
    output_batch = [data_loader.ids_to_tags(x) for x in np.argmax(output_batch, axis=-1)]

    return data_batch, output_batch, confidence_batch


def predict_from_workspace(workspace_dir, input_data):
    """
        Evaluate the model on the test set.
    """
    global args, data_loader

    data_dir = workspace_dir
    model_dir = os.path.join(data_dir, "model")

    # Load the parameters
    args = parser.parse_args()
    trgt_json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(trgt_json_path), "No json configuration file found at {}".format(trgt_json_path)

    params = utils.Params(trgt_json_path)
    params.data_dir = data_dir if data_dir else args.data_dir
    params.model_dir = model_dir if model_dir else args.model_dir

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(params.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    data_loader = DataLoader(params.data_dir, params)
    data = data_loader.load_data_for_predict(input_data)
    batch_sentences = data["predict"]["data"]

    # compute length of longest sentence in batch
    batch_max_len = max([len(s) for s in batch_sentences])

    # prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
    # initialising labels to -1 differentiates tokens with tags from PADding tokens
    batch_data = data_loader.pad_ind * np.ones((len(batch_sentences), batch_max_len))

    # copy the data to the numpy array
    for j in range(len(batch_sentences)):
        cur_len = len(batch_sentences[j])
        batch_data[j][:cur_len] = batch_sentences[j]

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    logging.info("Starting prediction")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    results = predict(model, batch_data)

    return results


if __name__ == '__main__':
    workspace_dir1 = "data/small"
    test_data = input_data = ["Sarin gas attacks on the Tokyo subway system in 1995 killed 12 people and injured thousands ."]

    data, output, confidence = predict_from_workspace(workspace_dir1, test_data)
    print("Train batch", data)
    print("Output batch", output)
    print("Output confidence", confidence)

    concat_word_piece = []

    for d, o, c in zip(data[0], output[0], confidence[0]):
        if d.startswith("##") and concat_word_piece:
            last_d, last_o, last_c = concat_word_piece[-1]
            last_d += d.replace("##", "")
        else:
            concat_word_piece.append((d, o, c))

    concepts = []
    for i in range(len(concat_word_piece)):
        d, o, c = concat_word_piece[i]
        if o.startswith("B-"):
            concepts.append((d, o.replace("B-", ""), c))
        elif o.startswith("I-"):
            if concepts:
                if concepts[0][-1] != o[2:]:
                    del concepts[-1]
                else:
                    ld, lo, lc = concepts[-1]
                    concepts[-1] = (ld, lo, (lc + c) / 2.0)

    print("Extract concepts:", concepts)

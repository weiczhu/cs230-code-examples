"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_sentences_from_text(path_txt, vocab):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_txt)

    # Convert line into list of tokens, splitting by white space
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # Lookup tokens to return their ids
    dataset = dataset.map(lambda tokens: {'sentences': vocab.lookup(tokens), 'sentence_lengths': tf.size(tokens)})

    return dataset


def load_labels_from_text(path_txt, vocab):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_txt)

    # Convert line into list of tokens, splitting by white space
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # Lookup tokens to return their ids
    dataset = dataset.map(lambda tokens: vocab.lookup(tokens))

    return dataset


def input_fn(mode, path_sentences, path_labels, params):
    """Input function for NER

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        sentences: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(params.path_words, num_oov_buckets=params.num_oov_buckets)
    tags = tf.contrib.lookup.index_table_from_file(params.path_tags)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, tf.tables_initializer(name='init_all_tables'))

    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    repeat_num = params.num_epochs if is_training else 1
    batch_size = params.batch_size if is_training else params.eval_size
    buffer_size = params.buffer_size if is_training else 1

    # Create the input data pipeline
    sentences = load_sentences_from_text(path_sentences, words)

    labels = load_labels_from_text(path_labels, tags)

    # Zip the sentence and the labels together
    dataset = tf.data.Dataset.zip((sentences, labels))

    id_pad_word = words.lookup(tf.constant(params.pad_word))
    id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create batches and pad the sentences of different length
    padded_shapes = ({'sentences': tf.TensorShape([None]),
                      'sentence_lengths': tf.TensorShape([])},
                     tf.TensorShape([None]))

    padding_values = ({'sentences': id_pad_word,
                       'sentence_lengths': 0},
                      id_pad_tag)

    dataset = (dataset
               .repeat(repeat_num)
               .shuffle(buffer_size=buffer_size)
               .padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
               .prefetch(1)
               )

    return dataset

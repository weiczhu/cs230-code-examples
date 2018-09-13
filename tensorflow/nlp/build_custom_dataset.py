#!/usr/bin/python2.7

import sre_yield
import numpy as np
import json
import os

file_path = "original_data.txt"
data_dir = "data/custom/"


def build_custom_dataset(in_file_path, data_dir):

    dataset_samples = []
    dataset_annotates = []

    dictionary = []
    in_file_path = os.path.join(data_dir, in_file_path)
    with open(in_file_path) as f:
        for line in f:
            j_content = json.loads(line)

            ken = j_content.get('ken')[0]
            city = j_content.get('city')[0]
            area = j_content.get('area')[0]
            zp = j_content.get('zipcode')[0]

            dictionary.append((ken.lower(), 'KEN'))
            dictionary.append((city.lower(), 'CIT'))
            dictionary.append((area.lower(), 'ARE'))
            dictionary.append((zp.lower(), 'ZIP'))

            ken = gen_annotate_entity(ken, 'KEN')
            city = gen_annotate_entity(city, 'CIT')
            area = gen_annotate_entity(area, 'ARE')
            zp = gen_annotate_entity(zp, 'ZIP')


            samples, annotates = make_sampels_and_annotates(ken, city, area, zp)

            # for sample, annotate in zip(samples, annotates):
            #     if sample not in dataset_samples:
            #         dataset_samples.append(sample)
            #         dataset_annotates.append(annotate)

            dataset_samples.extend(samples)
            dataset_annotates.extend(annotates)

    dataset = []
    for s, a in zip(dataset_samples, dataset_annotates):
        dataset.append((s, a))

    print(' dataset :', dataset[:10])
    dataset = list(set(dataset))
    np.random.shuffle(dataset)
    print(' dataset :', dataset[:10])
    dataset_samples, dataset_annotates = zip(*dataset)

    print('dataset_samples:', len(dataset_samples))
    print('dataset_annotates:', len(dataset_annotates))

    with open(data_dir + 'sentences.txt', 'w') as f:
        for item in dataset_samples:
            f.write("%s\n" % item.lower())

    with open(data_dir + 'labels.txt', 'w') as f:
        for item in dataset_annotates:
            f.write("%s\n" % item)

    splitidx = int(len(dataset_samples) * .9)
    train_samples, test_samples = dataset_samples[:splitidx], dataset_samples[splitidx:]
    train_annotates, test_annotates = dataset_annotates[:splitidx], dataset_annotates[splitidx:]

    with open(data_dir + 'train/sentences.txt', 'w') as f:
        for item in train_samples:
            f.write("%s\n" % item.lower())

    with open(data_dir + 'train/labels.txt', 'w') as f:
        for item in train_annotates:
            f.write("%s\n" % item)

    with open(data_dir + 'test/sentences.txt', 'w') as f:
        for item in test_samples:
            f.write("%s\n" % item.lower())

    with open(data_dir + 'test/labels.txt', 'w') as f:
        for item in test_annotates:
            f.write("%s\n" % item)

    with open(data_dir + 'dev/sentences.txt', 'w') as f:
        for item in test_samples:
            f.write("%s\n" % item.lower())

    with open(data_dir + 'dev/labels.txt', 'w') as f:
        for item in test_annotates:
            f.write("%s\n" % item)

    with open(data_dir + 'dictionary.txt', 'w') as f:
        for (word, type) in dictionary:
            f.write("%s::%s\n" % (word, type))

def make_sampels_and_annotates(ken, city, area, zp):

    geo = "({ken}|{city}|{area}|{area}, {city}, {ken} {zp})".format(ken=ken, city=city, area=area, zp=zp)

    # TBD, adding random samples, with multiple geo
    templates = [
    "I would like to go to {geo}".format(geo=geo),
    "I am at {geo}".format(geo=geo),
    "I want to go to {geo}".format(geo=geo),
    "My favorite place is {geo}".format(geo=geo),
    "What about {geo}".format(geo=geo),
    "I really like {geo}".format(geo=geo),
    "{geo} is a good place".format(geo=geo),
    "{geo} is a nice recommendation".format(geo=geo),
    "Where is {geo}".format(geo=geo),
    "{geo} is really nice".format(geo=geo),
    "I would like to make a reservation at {geo}".format(geo=geo),
    "My destination is {geo}".format(geo=geo),
    "I am heading to {geo}".format(geo=geo),
    "Could you help deliver it to {geo}".format(geo=geo),
    "Shall we go to {geo} for dinner".format(geo=geo),
    "I think {geo} is a good place for lunch".format(geo=geo),
    "Please help me find a restaurant at {geo}".format(geo=geo),
    ]

    templates = map(pre_process, templates)
    # print('templates:', templates)

    annotated_samples = []
    for template in templates:
        annotated_samples.extend(list(sre_yield.AllStrings(template)))
    # print(annotated_samples, len(annotated_samples))

    samples = []
    annotates = []
    for annotated_sample in annotated_samples:
        annotated_words = annotated_sample.split(" ")
        annotated_pairs = map(gen_annotate_pair, annotated_words)
        # print("here we are:", annotated_pairs, type(annotated_pairs))
        samplewords, annotatewords = zip(*annotated_pairs)

        samplesent = ' '.join(samplewords)
        annotatesent = ' '.join(annotatewords)
        samples.append(samplesent)
        annotates.append(annotatesent)

    # print(samples, len(samples))
    # print(annotates, len(annotates))

    return (samples, annotates)


def pre_process(sent):
    return sent.replace(',', '')

def gen_annotate_entity(entity, type='GEO'):
    words = entity.split(' ')
    words_annotated = []
    for idx, word in enumerate(words):
        if idx == 0:
            word = word + '/B-' + type
        else:
            word = word + '/I-' + type

        words_annotated.append(word)

    return ' '.join(words_annotated)

def gen_annotate_pair(word):
    if '/B-' in word or 'I-' in word:
        return word.split("/")
    else:
        return [word, 'O']

if __name__ == '__main__':
    build_custom_dataset(file_path, data_dir)

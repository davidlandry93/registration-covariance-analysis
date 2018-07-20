
import argparse
import json
import os
import numpy as np
import random
import re
import sys

import recova.util
from recova.util import eprint, bat_distance, to_upper_triangular
from recova.learning import model_factory
from recova.learning_dataset import filter_dataset
from recova.learning.preprocessing import preprocessing_factory

def model_from_file(path, algorithm):
    model = model_factory(algorithm)
    model.load_model(path)

    return model

def model_loader(learning_run):
    """
    From the dictionary output by a learning run,
    load a model.
    """
    return model_from_file(learning_run['model'], learning_run['metadata']['algorithm'])

def train_test_split_cross_validate(dataset, mask, cv_model):
    train_indices = []
    validation_indices = []

    for i in np.nonzero(mask)[0]:
        if dataset['data']['pairs'][i]['dataset'] == cv_model:
            validation_indices.append(int(i))
        else:
            train_indices.append(int(i))

    return train_indices, validation_indices

def train_test_split_validate_on_end(dataset, mask):
    train_indices = []
    validation_indices = []

    dict_of_locations = dict()
    for i in np.nonzero(mask)[0]:
        pair = dataset['data']['pairs'][i]
        if pair['dataset'] not in dict_of_locations:
            dict_of_locations[pair['dataset']] = pair['reference']
        else:
            old_value = dict_of_locations[pair['dataset']]
            dict_of_locations[pair['dataset']] = max(pair['reference'], old_value)

    print(dict_of_locations)

    for i in np.nonzero(mask)[0]:
        pair = dataset['data']['pairs'][i]
        if pair['reference'] < 0.7 * dict_of_locations[pair['dataset']]:
            train_indices.append(int(i)) # Convert from np.int64 to a regular int
        else:
            validation_indices.append(int(i))

    return train_indices, validation_indices

def train_test_split(dataset, mask):
    train_indices = []
    validation_indices = []

    idx = np.nonzero(mask)[0]
    random.shuffle(idx)

    print(idx)

    for i in range(int(0.7 * len(idx))):
        train_indices.append(int(idx[i]))
    for i in range(int(0.7 * len(idx)), len(idx)):
        validation_indices.append(int(idx[i]))

    return train_indices, validation_indices

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str)
    parser.add_argument('output', type=str, help='Where to save the learning run')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-7)
    parser.add_argument('-a', '--alpha', type=float, default=1e-6)
    parser.add_argument('-b', '--beta', type=float, default=1e3)
    parser.add_argument('-n', '--n_iterations', type=int, default=0, help='Maximum number of iterations. 0 means no maximum and wait for convergence.')
    parser.add_argument('-pa', '--patience', type=int, default=20, help='N of iterations without improvement before ending training.')
    parser.add_argument('-cv', '--cross-validate', type=str, help='Name of the dataset to use as a validation set', default='')
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-10, help='For the MLP, set the weight decay parameter.')
    parser.add_argument('--filter', type=str, help='Filter out datasets from the learning.', default='')
    parser.add_argument('--preprocessing', '-p', type=str, help='Name of the preprocessing algorithm to use.', default='identity')
    parser.add_argument('-md', '--min-delta', type=float, help='Minimum gain on the validation loss before the learning stops.', default=1e-4)
    parser.add_argument('--validate-on-end', action='store_true', help='Train on the first part of the dataset, validate on the second part.')
    args = parser.parse_args()

    eprint('Loading document')
    input_document = json.load(sys.stdin)
    eprint('Done loading document')

    if args.filter:
        regex = re.compile(args.filter)
        mask = filter_dataset(input_document, regex)
    else:
        mask = np.ones(len(input_document['data']['pairs']), dtype=bool)

    print(np.sum(mask))


    train_indices = []
    validation_indices = []
    if args.cross_validate:
        train_indices, validation_indices = train_test_split_cross_validate(input_document, mask, args.cross_validate)
    elif args.validate_on_end:
        train_indices, validation_indices = train_test_split_validate_on_end(input_document, mask)
    else:
        train_indices, validation_indices = train_test_split(input_document, mask)

    eprint('Training set size: {}. Validation set size: {}'.format(len(train_indices), len(validation_indices)))
    eprint(type(train_indices[0]))

    predictors = np.array(input_document['data']['xs'])
    covariances = np.array(input_document['data']['ys'])


    model = model_factory(args.algorithm)

    model.learning_rate = args.learning_rate
    model.alpha = args.alpha
    model.beta = args.beta
    model.n_iterations = args.n_iterations
    model.weight_decay = args.weight_decay
    model.min_delta = args.min_delta
    model.patience = args.patience

    preprocessing_algo = preprocessing_factory(args.preprocessing)
    model.preprocessing = preprocessing_algo

    if train_indices and validation_indices:
        learning_run = model.fit(predictors, covariances, train_set=train_indices, test_set=validation_indices)
    else:
        learning_run = model.fit(predictors, covariances)

    learning_run['metadata']['descriptor_config'] = input_document['metadata']['descriptor_config']

    if args.cross_validate:
        learning_run['metadata']['cross_validation'] = args.cross_validate

    model_path = args.output + '.model'
    model.save_model(model_path)
    learning_run['model'] = os.getcwd() + '/' + model_path


    for key in learning_run:
        print(key)
        print(learning_run[key])
        json.dumps(learning_run[key])
        print()
        print()

    with open(args.output + '.json', 'w') as f:
        json.dump(learning_run, f)



if __name__ == '__main__':
    cli()


import argparse
import json
import os
import numpy as np
import sys

import recova.util
from recova.util import eprint, bat_distance, to_upper_triangular
from recova.learning import model_factory

def model_loader(learning_run):
    """
    From the dictionary output by a learning run,
    load a model.
    """
    model = model_factory(learning_run['metadata']['algorithm'])
    model.load_model(learning_run['model'])

    return model

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-a', '--alpha', type=float, default=1e-4)
    parser.add_argument('-b', '--beta', type=float, default=1e3)
    parser.add_argument('-n', '--n_iterations', type=int, default=0, help='Maximum number of iterations. 0 means no maximum and wait for convergence.')
    parser.add_argument('-w', '--convergence_window', type=int, default=20, help='N of iterations without improvement before ending training.')
    parser.add_argument('-cv', '--cross-validate', type=str, help='Name of the dataset to use as a validation set', default='')
    parser.add_argument('-o', '--output', type=str, help='Where to save the learning run')
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-10, help='For the MLP, set the weight decay parameter.')
    args = parser.parse_args()

    eprint('Loading document')
    input_document = json.load(sys.stdin)
    eprint('Done loading document')

    sys.stdin = open('/dev/tty')

    predictors = np.array(input_document['data']['xs'])
    covariances = np.array(input_document['data']['ys'])

    train_indices = []
    validation_indices = []
    if args.cross_validate:
        for i, pair in enumerate(input_document['data']['pairs']):
            if pair['dataset'] == args.cross_validate:
                validation_indices.append(i)
            else:
                train_indices.append(i)

        eprint('Training set size: {}. Validation set size: {}'.format(len(train_indices), len(validation_indices)))

    model = model_factory(args.algorithm)

    model.learning_rate = args.learning_rate
    model.alpha = args.alpha
    model.beta = args.beta
    model.n_iterations = args.n_iterations
    model.convergence_window = args.convergence_window
    model.weight_decay = args.weight_decay

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

    for k in learning_run:
        eprint('{}: {}'.format(k, learning_run[k]))
        eprint()

    with open(args.output + '.json', 'w') as f:
        json.dump(learning_run, f)



if __name__ == '__main__':
    cli()

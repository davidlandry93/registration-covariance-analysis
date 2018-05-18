
import argparse
import json
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
    model.import_model(learning_run['model'])

    return model

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-a', '--alpha', type=float, default=1e-4)
    parser.add_argument('-b', '--beta', type=float, default=1e3)
    parser.add_argument('-n', '--n_iterations', type=int, default=200)
    parser.add_argument('-w', '--convergence_window', type=int, default=20, help='N of iterations without improvement before ending training.')
    args = parser.parse_args()

    eprint('Loading document')
    input_document = json.load(sys.stdin)
    eprint('Done loading document')

    sys.stdin = open('/dev/tty')

    predictors = np.array(input_document['data']['xs'])

    covariances = np.array(input_document['data']['ys'])

    model = model_factory(args.algorithm)

    model.learning_rate = args.learning_rate
    model.alpha = args.alpha
    model.beta = args.beta
    model.n_iterations = args.n_iterations
    model.convergence_window = args.convergence_window

    learning_run = model.fit(predictors, covariances)

    json.dump(learning_run, sys.stdout)


if __name__ == '__main__':
    cli()

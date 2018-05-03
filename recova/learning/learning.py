
import argparse
import json
import numpy as np
import sys

import recova.util
from recova.util import eprint, bat_distance
from recova.learning import model_factory


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str)
    args = parser.parse_args()

    eprint('Loading document')
    input_document = json.load(sys.stdin)
    eprint('Done loading document')

    sys.stdin = open('/dev/tty')

    predictors = np.array(input_document['data']['predictors'])

    np_examples = []
    covariances = np.empty((len(predictors), 6, 6))
    for i, example_batch in enumerate(input_document['data']['errors']):
        errors = np.array(example_batch)
        covariances[i,:,:] = np.dot(errors.T, errors)

    model = model_factory(args.algorithm)

    learning_run = model.fit(predictors, covariances)
    print(learning_run)


if __name__ == '__main__':
    cli()

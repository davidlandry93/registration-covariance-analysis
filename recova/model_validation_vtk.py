
import argparse
import json
import numpy as np
import time

from recova.learning.learning import model_loader
from recova.distribution_to_vtk_ellipsoid import distribution_to_vtk_ellipsoid
from recova.util import eprint

def frobenius(tensor):
    print(tensor.shape)
    tensor = np.power(tensor, 2.0)

    print(tensor.shape)
    tensor = np.sum(tensor, axis=2)

    print(tensor.shape)
    tensor = np.sum(tensor, axis=1)
    
    print(tensor.shape)
    return np.sqrt(tensor)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to the dataset used to train the model', type=str)
    parser.add_argument('model', help='Path to the trained model', type=str)
    parser.add_argument('output', help='Where to output the vtk files', type=str)
    args = parser.parse_args()

    print('Loading dataset...')
    with open(args.dataset) as f:
        dataset = json.load(f)
    print('Done')

    print('Loading model...')
    with open(args.model) as f:
        learning_run = json.load(f)
    print('Done')


    model = model_loader(learning_run)

    xs = np.array(dataset['data']['xs'])
    ys = np.array(dataset['data']['ys'])

    xs_validation = xs[learning_run['validation_set']]
    ys_validation = ys[learning_run['validation_set']]

    pred_begin = time.time()
    ys_predicted = model.predict(xs_validation)
    print((time.time() - pred_begin) / len(xs))

    errors = frobenius(ys_validation - ys_predicted)
    print(np.mean(errors))

    for i in range(len(ys_predicted)):
        distribution_to_vtk_ellipsoid(np.zeros(3), ys_validation[i][0:3,0:3], args.output + '_translation_validation_' + str(i).zfill(4))
        distribution_to_vtk_ellipsoid(np.zeros(3), ys_predicted[i][0:3,0:3], args.output + '_translation_predicted_' + str(i).zfill(4))


        distribution_to_vtk_ellipsoid(np.zeros(3), ys_validation[i][3:6,3:6], args.output + '_rotation_validation_' + str(i).zfill(4))
        distribution_to_vtk_ellipsoid(np.zeros(3), ys_predicted[i][3:6,3:6], args.output + '_rotation_predicted_' + str(i).zfill(4))




if __name__ == '__main__':
    cli()


import argparse
import csv
import json
import numpy as np
import re
import time

from recov.pointcloud_io import pointcloud_to_vtk

from recova.learning.learning import model_from_file
from recova.distribution_to_vtk_ellipsoid import distribution_to_vtk_ellipsoid
from recova.registration_result_database import RegistrationPairDatabase
from recova.util import eprint, kullback_leibler, parallel_starmap_progressbar

def frobenius(tensor):
    print(tensor.shape)
    tensor = np.power(tensor, 2.0)

    print(tensor.shape)
    tensor = np.sum(tensor, axis=2)

    print(tensor.shape)
    tensor = np.sum(tensor, axis=1)

    print(tensor.shape)
    return np.sqrt(tensor)

def generate_one_prediction(i, y_predicted, pair_id, registration_pair_database, output):
    distribution_to_vtk_ellipsoid(np.zeros(3), y_predicted[0:3,0:3], output + '/translation_predicted_' + str(i).zfill(4))

    distribution_to_vtk_ellipsoid(np.zeros(3), y_predicted[3:6,3:6], output + '/rotation_predicted_' + str(i).zfill(4))

    registration_pair = registration_pair_database.get_registration_pair(pair_id['dataset'], pair_id['reading'], pair_id['reference'])

    reading = registration_pair.points_of_reading()

    pointcloud_to_vtk(reading, output + '/reading_{}'.format(str(i).zfill(4)))


def prediction_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to the dataset used to train the model', type=str)
    parser.add_argument('model', help='Path to the trained model', type=str)
    parser.add_argument('output', help='Where to output the vtk files', type=str)
    parser.add_argument('--registration-database', help='Fetch the pointclouds to give some context to the generated covariances.')
    parser.add_argument('--filter', help='Locations to filter during the query', type=str, default='')
    args = parser.parse_args()

    print('Loading dataset...')
    with open(args.dataset) as f:
        dataset = json.load(f)
    print('Done')

    filtering_re = re.compile(args.filter)


    model = model_from_file(args.model, 'cello')

    eprint(model)

    xs = np.array(dataset['data']['xs'])

    pairs = dataset['data']['pairs']
    selection = np.ones(len(pairs), dtype=np.bool)
    for i, pair in enumerate(pairs):
        if filtering_re.match(pair['dataset']) and args.filter:
            selection[i] = 0

    eprint(len(selection))
    eprint(selection.sum())

    xs = xs[selection]

    ys_predicted = model.predict(xs)
    np.save(args.output + '/predictions.npy', ys_predicted)

    db = RegistrationPairDatabase(args.registration_database)

    parallel_starmap_progressbar(generate_one_prediction, [(i, ys_predicted[i], dataset['data']['pairs'][i], db, args.output) for i in range(len(ys_predicted))])


    # for i in range(len(ys_predicted)):
    #     distribution_to_vtk_ellipsoid(np.zeros(3), ys_predicted[i][0:3,0:3], args.output + '/translation_predicted_' + str(i).zfill(4))

    #     distribution_to_vtk_ellipsoid(np.zeros(3), ys_predicted[i][3:6,3:6], args.output + '/rotation_predicted_' + str(i).zfill(4))

    #     pair_id = dataset['data']['pairs'][i]
    #     registration_pair = db.get_registration_pair(pair_id['dataset'], pair_id['reading'], pair_id['reference'])

    #     reading = registration_pair.points_of_reading()

    #     pointcloud_to_vtk(reading, args.output + '/reading_{}'.format(str(i).zfill(4)))


def dataset_to_vtk_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to learning set', type=str)
    parser.add_argument('output', help='Output path for vtk files', type=str)
    args = parser.parse_args()

    with open(args.dataset) as f:
        pass




def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to the dataset used to train the model', type=str)
    parser.add_argument('learningrun', help='Path to the learning run', type=str)
    parser.add_argument('model', help='Path to the trained model', type=str)
    parser.add_argument('output', help='Where to output the vtk files', type=str)
    args = parser.parse_args()

    print('Loading dataset...')
    with open(args.dataset) as f:
        dataset = json.load(f)
    print('Done')

    print('Loading model...')
    with open(args.learningrun) as f:
        learning_run = json.load(f)
    print('Done')


    model = model_from_file(args.model, learning_run['metadata']['algorithm'])

    xs = np.array(dataset['data']['xs'])
    ys = np.array(dataset['data']['ys'])

    xs_validation = xs[learning_run['validation_set']]
    ys_validation = ys[learning_run['validation_set']]

    pred_begin = time.time()
    ys_predicted = model.predict(xs_validation)
    print((time.time() - pred_begin) / len(xs))

    errors = frobenius(ys_validation - ys_predicted)

    klls = []
    for i in range(len(ys_validation)):
        kll_left = kullback_leibler(ys_validation[i], ys_predicted[i])
        klls.append(kll_left)

    print(np.mean(errors))
    print(np.mean(np.array(klls)))

    with open(args.output + '/summary.csv', 'w') as summary_file:
        writer = csv.DictWriter(summary_file, ['location', 'reading', 'reference', 'loss', 'kullback_leibler', 'predicted_trace', 'reference_trace'])
        writer.writeheader()

        for i in range(len(ys_predicted)):
            distribution_to_vtk_ellipsoid(np.zeros(3), ys_validation[i][0:3,0:3], args.output + '/translation_validation_' + str(i).zfill(4))
            distribution_to_vtk_ellipsoid(np.zeros(3), ys_predicted[i][0:3,0:3], args.output + '/translation_predicted_' + str(i).zfill(4))


            distribution_to_vtk_ellipsoid(np.zeros(3), ys_validation[i][3:6,3:6], args.output + '/rotation_validation_' + str(i).zfill(4))
            distribution_to_vtk_ellipsoid(np.zeros(3), ys_predicted[i][3:6,3:6], args.output + '/rotation_predicted_' + str(i).zfill(4))

            eprint(learning_run['validation_set'][i])
            index_of_example = learning_run['validation_set'][i]
            writer.writerow({
                'location': dataset['data']['pairs'][index_of_example]['dataset'],
                'reading': dataset['data']['pairs'][index_of_example]['reading'],
                'reference': dataset['data']['pairs'][index_of_example]['reference'],
                'loss': errors[i],
                'kullback_leibler': klls[i],
                'predicted_trace': np.trace(ys_predicted[i]),
                'reference_trace': np.trace(ys_validation[i])
            })



if __name__ == '__main__':
    cli()

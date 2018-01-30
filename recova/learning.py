
import argparse
import json
import os
import numpy as np
import sys

from sklearn.neighbors import DistanceMetric, BallTree


class CelloModel:
    BALL_SIZE = 100.

    def __init__(self, predictors, errors, parameters):
        up = vector_to_upper_triangular(parameters)
        metric_matrix = np.dot(up.T, up)
        metric = DistanceMetric.get_metric('mahalanobis', VI=metric_matrix)

        self.errors = errors
        self.tree = BallTree(predictors, metric=metric)

    def metric_damping_function(self, metric_value):
        return max(200. - metric_value, 0.)

    def query(self, point):
        indices, distances = self.tree.query_radius([point], r=self.BALL_SIZE, return_distance=True)

        print(indices)
        print(distances)

        sum_of_damped_distances = 0.
        covariance = np.zeros((6,6))

        for i in range(len(indices)):
            # Disallow self matches.
            if distances[i] == 0.:
                continue

            # Compute the weight of one error vector associated with this descriptor.
            weight = self.metric_damping_function(distances[i])
            sum_of_damped_distances += weight * len(self.errors[i])

            for matched_predictor in indices[i]:
                errors_of_predictor = self.errors[matched_predictor]

                for e in errors_of_predictor:
                    e = e[np.newaxis]
                    print(np.dot(e.T, e))
                    covariance += weight * np.dot(e.T, e)

        covariance /= sum_of_damped_distances

        return covariance


def size_of_vector(n):
    """The size of a vector representing an nxn upper triangular matrix."""
    return int((n * n + n) / 2.) + 1

def upper_triangular_to_vector(up):
    pass

def vector_to_upper_triangular(vector):
    # Infer the shape of the matrix from the size of the vector.
    y = len(vector)
    n = int((-1 + np.sqrt(8. * y)) / 2.)

    matrix = np.zeros((n,n))
    cursor = 0

    for i in range(n):
        for j in range(i, n):
            matrix[i,j] = vector[cursor]
            cursor = cursor + 1

    return matrix


def compute_loss(model, predictors, errors):
    loss = 0.

    for i in range(len(predictors)):
        print(i)
        predicted_cov = model.query(predictors[i])

        print('Predicted covariance: ')
        print(predicted_cov)

        # First term of the loss. See Cello eq. 29.
        loss += len(errors[i]) * np.linalg.norm(predicted_cov)

        inv_predicted_cov = np.linalg.inv(predicted_cov)

        for e in errors[i]:
            loss += np.dot(e, np.dot(inv_predicted_cov, e))

    return loss


def cello_learning_cli():
    print('Loading document')
    input_document = json.load(sys.stdin)
    print('Done loading document')

    predictors = np.array(input_document['data']['predictors'])

    np_examples = []
    for example_batch in input_document['data']['errors']:
        np_examples.append(np.array(example_batch))

    print(predictors.shape)
    print(np_examples[0].shape)

    size_of_predictor = predictors.shape[1]
    sz_of_vector = size_of_vector(size_of_predictor)

    model = CelloModel(predictors[0:300], np_examples[0:300], np.ones(sz_of_vector))

    loss = compute_loss(model, predictors[300:], np_examples[300:])
    print(loss)

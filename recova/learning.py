
import argparse
import json
import os
import numpy as np
import sys

from sklearn.neighbors import DistanceMetric, BallTree


class CelloModel:
    def __init__(self, predictors, errors, parameters):
        up = vector_to_upper_triangular(parameters)
        metric_matrix = np.dot(up.T, up)
        metric = DistanceMetric.get_metric('mahalanobis', VI=metric_matrix)

        self.errors = errors
        self.tree = BallTree(predictors, metric=metric)

    def query(self, point):
        indices, distances = self.tree.query([point], k=10)
        print(indices)
        print(distances)

        return np.identity(6)

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

        # First term of the loss. See Cello eq. 29.
        loss += len(errors[i]) * np.linalg.norm(predicted_cov)

        inv_predicted_cov = np.linalg.inv(predicted_cov)

        for e in errors[i]:
            loss += np.dot(e, np.dot(inv_predicted_cov, e))

    return loss


def censi_learning_cli():
    input_document = json.load(sys.stdin)

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

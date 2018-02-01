
import argparse
import json
import os
from math import sqrt, ceil
import numpy as np
import sys

import torch
from torch.autograd import Variable

from sklearn.neighbors import DistanceMetric, BallTree

def to_upper_triangular(v):
    # Infer the size of the matrix from the size of the vector.
    n = ceil((-1 + sqrt(8. * len(v))) / 2.)

    # Create the list of indices to gather from the vector.
    rng = torch.arange(n)
    triangular_numbers = rng * (rng + 1) / 2

    col = n * torch.arange(n) - triangular_numbers
    row = rng
    index_matrix = col.view(n,1) + row.view(1,n)
    index_vector = index_matrix.view(index_matrix.numel())
    index_vector = Variable(index_vector.long())

    gathered = torch.gather(v, 0, index_vector)

    return gathered.view(n,n) * Variable(upper_triangular_mask(n).float())

def upper_triangular_to_vector(up):
    return up[upper_triangular_mask(up.size(-1))]


def upper_triangular_mask(n):
    v = torch.arange(n)
    return v >= v.view(n,1)


class NearestNeighbor(torch.autograd.Function):
    BALL_SIZE = 500.

    def __init__(self, predictors):
        self.predictors = predictors

    @staticmethod
    def forward(ctx, metric_matrix, predictor):
        metric = DistanceMetric.get_metric('mahalanobis', VI=metric_matrix)

        tree = BallTree(self.predictors, metric=metric)
        indices, distances = tree.query_radius([predictor], r=self.BALL_SIZE, return_distances=True)
        indices = indices[0]
        distances = distances[0]

        ctx.save_for_backward(metric_matrix, predictor, indices[0])

        all_distances = self.BALL_SIZE * torch.ones(self.predictors.size(0))
        all_distances[indices] = distances

        return all_distances

    @staticmethod
    def backward(ctx, distances_grad):
        metric_matrix, predictor, neighbors = ctx.saved_variables
        sum_of_predictor = torch.sum(predictor)

        return (torch.sum(self.predictors * sum_of_predictor, 1), torch.sum(torch.dot(self.predictors, metric_matrix), 1))


class CelloFunction(torch.autograd.Function):
    BALL_SIZE = 500.

    def __init__(self, predictors, errors):
        self.predictors = predictors
        self.errors = errors

        for error_collection in self.errors:
            self.covariance_of_predictors.append(np.dot(error_collection.T, error_collection))

    @staticmethod
    def forward(ctx, parameters, predictor):

        up = vector_to_upper_triangular(parameters)
        metric_matrix = np.dot(up.T, up)
        metric = DistanceMetric.get_metrix('mahalanobis', VI=metric_matrix)

        tree = BallTree(self.predictors, metric=metric)
        indices, distances = tree.query_radius([predictor], r=self.BALL_SIZE, return_distance=True)

        if len(indices) == 0:
            raise RuntimeError('No predictor was close enough to the query point.')

        sum_of_damped_distances = 0.
        if i in range(len(indices)):
            weight = self.metric_damping_function(distances[i])
            sum_of_damped_distances += weight * len(self.errors[i])

            covariance += weight * self.covariance_of_predictors[indices[i]]

        ctx.save_for_backward(predictor, distances)

        return covariance / sum_of_damped_distances

    @staticmethod
    def backward(ctx, grad_output):
        predictor, distances = ctx.saved_variables

        grad_of_k = [-1. if rho < self.BALL_SIZE else 0. for rho in distances]


        return parameters_grad, predictor_grad


class CelloModel:
    BALL_SIZE = 500.

    def __init__(self, predictors, errors, parameters):
        up = vector_to_upper_triangular(parameters)
        metric_matrix = np.dot(up.T, up)
        metric = DistanceMetric.get_metric('mahalanobis', VI=metric_matrix)

        self.errors = errors
        self.tree = BallTree(predictors, metric=metric)

        self.covariance_of_predictors = []

        print('Precomputing covariances')


    def metric_damping_function(self, metric_value):
        return max(200. - metric_value, 0.)


    def query(self, point):
        indices, distances = self.tree.query_radius([point], r=self.BALL_SIZE, return_distance=True)
        indices = indices[0]
        distances = distances[0]

        sum_of_damped_distances = 0.
        covariance = np.zeros((6,6))

        if len(indices) == 0:
            raise RuntimeError('No predictor was close enough to the query point.')

        for i in range(len(indices)):
            # Disallow self matches.
            if distances[i] == 0.:
                continue

            # Compute the weight of one error vector associated with this descriptor.
            weight = self.metric_damping_function(distances[i])
            sum_of_damped_distances += weight * len(self.errors[i])

            covariance += weight * self.covariance_of_predictors[indices[i]]

        covariance /= sum_of_damped_distances

        return covariance


def size_of_vector(n):
    """The size of a vector representing an nxn upper triangular matrix."""
    return int((n * n + n) / 2.)

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
        predicted_cov = model.query(predictors[i])

        # First term of the loss. See Cello eq. 29.
        loss += len(errors[i]) * np.linalg.norm(predicted_cov)

        inv_predicted_cov = np.linalg.inv(predicted_cov)

        # Use matrix operations to compute the second term of the loss for multiple error vectors.
        err_losses = np.dot(errors[i], inv_predicted_cov)
        err_losses = err_losses * errors[i] # Term by term multiplication.
        err_losses = np.sum(err_losses)

        loss += err_losses

    return loss

def cello_learning(predictors, errors):
    size_of_predictor = predictors.shape[1]
    sz_of_vector = size_of_vector(size_of_predictor)

    model = CelloModel(predictors[0:300], np_examples[0:300], np.ones(sz_of_vector))

    loss = compute_loss(model, predictors[300:], np_examples[300:])
    print(loss)


def compute_distances(predictors, metric_matrix, predictor):
    delta = predictors - predictor
    lhs = torch.mm(delta, metric_matrix)
    return torch.sum(lhs * delta, 1).squeeze()


def predict(predictors, covariances, distances, predictor):
    zero_distances = distances < 1e-10
    distances.masked_fill(zero_distances, 1.)

    weights = torch.clamp(1. - distances, min=0.)

    predicted_cov = torch.sum(covariances * weights.view(-1,1,1), 0) / torch.sum(weights)

    return predicted_cov


def kullback_leibler(cov1, cov2):
    """Returns the kullback leibler divergence on a pair of covariances that have the same mean.
    See http://bit.ly/2FAYCgu."""
    det1, det2 = np.linalg.det(cov1.data), np.linalg.det(cov2.data)

    print('KLL')
    print(cov1.data)
    print(det1)
    print(cov2.data)
    print(det2)

    A = torch.trace(torch.mm(torch.inverse(cov1), cov2))
    B = 6.
    C = float(np.log(det1 / det2))
    return 0.5 * (A - B + C)


def loss_of_covariance(lhs, rhs):
    return torch.log(torch.norm(torch.mm(torch.inverse(lhs), rhs) - Variable(torch.eye(6))))


def loss_of_set(xs, ys, metric, test_xs, test_ys):
    sum_of_losses = 0.
    for i, x in enumerate(test_xs):
        distances = compute_distances(xs, metric, x)
        predicted_cov = predict(xs, ys, distances, x)
        sum_of_losses += kullback_leibler(test_ys[i], predicted_cov)

    return sum_of_losses / len(test_xs)


def theta_to_metric_matrix(theta):
    up = to_upper_triangular(theta)
    return torch.mm(up, up.transpose(0,1))


def cello_torch(predictors, covariances):
    alpha = 1e-5

    size_of_predictor = predictors.shape[1]
    print('Size of predictor: {}'.format(size_of_predictor))
    sz_of_vector = size_of_vector(size_of_predictor)

    idx = np.arange(len(predictors))
    np.random.shuffle(idx)

    training_set_size = int(len(predictors) * 0.8)

    predictors_training = Variable(torch.Tensor(predictors[idx[0:training_set_size]]))
    covariances_training = Variable(torch.Tensor(covariances[idx[0:training_set_size]]))

    predictors_validation = Variable(torch.Tensor(predictors[idx[training_set_size:]]))
    covariances_validation = Variable(torch.Tensor(covariances[idx[training_set_size:]]))

    theta = Variable(torch.randn(sz_of_vector) / 1000., requires_grad=True)

    for epoch in range(500):
        for i, predictor in enumerate(predictors_training):
            metric_matrix = theta_to_metric_matrix(theta)

            distances = compute_distances(predictors_training, metric_matrix, predictor)
            predicted_cov = predict(predictors_training, covariances_training, distances, predictor)

            loss_lhs = torch.log(torch.norm(predicted_cov))
            loss_rhs = loss_of_covariance(covariances_training[i], predicted_cov)

            nonzero_distances = torch.gather(distances, 0, torch.nonzero(distances).squeeze())

            regularization_term = torch.sum(torch.log(nonzero_distances))
            loss = (1 - alpha) * (loss_lhs + loss_rhs ) +  alpha * regularization_term

            # print('Norm: {}, Cov: {}, Reg: {}'.format(loss_lhs.data[0], loss_rhs.data[0], regularization_term.data[0]))

            loss.backward(retain_graph=True)

            theta.data -= 1e-5 * theta.grad.data
            theta.grad.data.zero_()

        print('VALIDATION')
        print(theta)
        print(loss_of_set(predictors_training, covariances_training, metric_matrix, predictors_validation, covariances_validation))


def cello_learning_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str)
    args = parser.parse_args()

    print('Loading document')
    input_document = json.load(sys.stdin)
    print('Done loading document')

    predictors = np.array(input_document['data']['predictors'])

    np_examples = []
    covariances = np.empty((len(predictors), 6, 6))
    for i, example_batch in enumerate(input_document['data']['errors']):
        errors = np.array(example_batch)
        covariances[i,:,:] = np.dot(errors.T, errors)

    if args.algorithm == 'cello':
        cello_learninig(predictors, np_examples)
    elif args.algorithm == 'cellotorch':
        cello_torch(predictors, covariances)



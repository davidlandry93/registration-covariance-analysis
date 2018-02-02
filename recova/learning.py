
import argparse
import json
import os
from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np
import sys

import sklearn
import sklearn.model_selection
import torch
import torch.optim as optim
from torch.autograd import Variable

from sklearn.neighbors import DistanceMetric, BallTree, KDTree

from recova.util import nearestPD, eprint


def kullback_leibler(cov1, cov2):
    """Returns the kullback leibler divergence on a pair of covariances that have the same mean.
    cov1 and cov2 must be numpy matrices.
    See http://bit.ly/2FAYCgu."""

    corrected_cov1, corrected_cov2 = nearestPD(cov1), nearestPD(cov2)
    det1, det2 = np.linalg.det(corrected_cov1), np.linalg.det(corrected_cov2)

    A = np.trace(np.dot(np.linalg.inv(corrected_cov1), corrected_cov2))
    B = 6.
    C = float(np.log(det1) - np.log(det2))

    kll = 0.5 * (A - B + C)

    return kll


class CovarianceEstimationModel:
    def fit(self, xs, ys):
        raise NotImplementedError('CovarianceEstimationModels must implement fit method')

    def predict(self, xs):
        raise NotImplementedError('CovarianceEstimationModels must implement predict method')

    def validate(self, xs, ys):
        """Given a validation set, outputs a loss."""
        predictions = self.predict(xs)

        total_loss = 0.
        for i in range(len(predictions)):
            total_loss += kullback_leibler(ys[i], predictions[i])

        return total_loss / len(xs)


class KnnModel(CovarianceEstimationModel):
    def __init__(self, k=12):
        self.default_k = k

    def fit(self, xs, ys):
        self.kdtree = KDTree(xs)
        self.examples = ys

    def predict(self, xs, p_k=None):
        k = self.default_k if p_k is None else p_k

        distances, indices = self.kdtree.query(xs, k=k)
        predictions = np.zeros((len(xs), 6, 6))

        for i in range(len(xs)):
            exp_dists = np.exp(-distances[i])
            sum_dists = np.sum(exp_dists)
            ratios = exp_dists / sum_dists

            for j in range(len(indices)):
                predicted = np.sum(self.examples[indices[i]] * ratios.reshape(k,1,1), axis=0)
                predictions[i] = predicted

        return predictions


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


def size_of_vector(n):
    """The size of a vector representing an nxn upper triangular matrix."""
    return int((n * n + n) / 2.)







class CelloCovarianceEstimationModel(CovarianceEstimationModel):
    def __init__(self, alpha=1e-6):
        self.alpha = alpha

    def fit(self, predictors, covariances):
        self.predictors = torch.Tensor(predictors)
        self.covariances = torch.Tensor(covariances)

        # Initialize a distance metric.
        sz_of_vector = size_of_vector(predictors.shape[1])
        self.theta = Variable(torch.randn(sz_of_vector) / 1000., requires_grad=True)

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=100)

        # optimizer = optim.SGD([theta], lr=1e-7)
        # optimizer = optim.Adam([theta], lr=1e-5)
        optimizer = optim.RMSprop([self.theta], lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

        for epoch, (train_set, test_set) in enumerate(selector.split(predictors)):
            optimizer.zero_grad()

            xs_train, xs_validation = Variable(torch.Tensor(predictors[train_set])), Variable(torch.Tensor(predictors[test_set]))
            ys_train, ys_validation = Variable(torch.Tensor(covariances[train_set])), Variable(torch.Tensor(predictors[test_set]))

            sum_of_losses = Variable(torch.Tensor([0.]))

            for i, x in enumerate(xs_train):
                metric_matrix = self.theta_to_metric_matrix(self.theta)

                distances = self.compute_distances(xs_train, metric_matrix, x)
                prediction = self.prediction_from_distances(ys_train, distances, x)

                loss_A = torch.log(torch.norm(prediction))
                loss_B = torch.log(torch.norm(torch.mm(torch.inverse(prediction), ys_train[i]) - Variable(torch.eye(6))))

                nonzero_distances = torch.gather(distances, 0, torch.nonzero(distances).squeeze())
                regularization_term = torch.sum(torch.log(nonzero_distances))

                optimization_loss = (1 - self.alpha) * (loss_A + loss_B) + self.alpha * regularization_term
                sum_of_losses += optimization_loss
                optimization_loss.backward()
                optimizer.step()

            average_loss = sum_of_losses / len(xs_train)

            print('Avg Optimization Loss: %f' % average_loss)
            print(epoch)
            if epoch % 10 == 0:
                print(self.validate(xs_validation, ys_validation))


    def predict(self, queries):
        metric_matrix = self.theta_to_metric_matrix(self.theta)

        print(metric_matrix)

        predictions = torch.zeros(len(queries),6,6)

        for i, x in enumerate(queries):
            print('PREDICTING %d' %i)
            distances = self.compute_distances(self.predictors, metric_matrix, x)
            predictions[i] = self.prediction_from_distances(self.covariances, distances, x)
            print(predictions[i])


        return predictions


    def theta_to_metric_matrix(self, theta):
        up = to_upper_triangular(theta)
        return torch.mm(up, up.transpose(0,1))


    def compute_distances(self, predictors, metric_matrix, predictor):
        print(predictors.shape)
        print(metric_matrix.shape)
        print(predictor.shape)
        delta = predictors - predictor
        lhs = torch.mm(delta, metric_matrix)
        return torch.sum(lhs * delta, 1).squeeze()


    def prediction_from_distances(self, covariances, distances, x):
        zero_distances = distances < 1e-10
        distances.masked_fill(zero_distances, 1.)

        weights = torch.clamp(1. - distances, min=0.)
        predicted_cov = torch.sum(covariances * weights.view(-1,1,1), 0) / torch.sum(weights)

        return predicted_cov




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

    optimizer = optim.SGD([theta], lr=1e-5)

    for epoch in range(500):
        for i, predictor in enumerate(predictors_training):
            optimizer.zero_grad()
            metric_matrix = theta_to_metric_matrix(theta)

            distances = compute_distances(predictors_training, metric_matrix, predictor)
            predicted_cov = predict(predictors_training, covariances_training, distances, predictor)

            loss_lhs = torch.log(torch.norm(predicted_cov))
            loss_rhs = loss_of_covariance(covariances_training[i], predicted_cov)

            nonzero_distances = torch.gather(distances, 0, torch.nonzero(distances).squeeze())

            regularization_term = torch.sum(torch.log(nonzero_distances))
            loss = (1 - alpha) * (loss_lhs + loss_rhs ) +  alpha * regularization_term

            loss.backward(retain_graph=True)
            optimizer.step()


        print('VALIDATION')
        print(loss_of_set(predictors_training, covariances_training, metric_matrix, predictors_validation, covariances_validation))


def covariance_model_performance(model, predictors, covariances, p_selection=None):
    scores = []

    selection = (sklearn.model_selection.ShuffleSplit(n_splits = 100, test_size=0.25) if
                 p_selection is None else p_selection)

    for training_set, test_set in selection.split(predictors, covariances):
        model.fit(predictors[training_set], covariances[training_set])
        score = model.validate(predictors[test_set], covariances[test_set])
        scores.append(score)


    scores = np.array(scores)
    return scores.mean(), scores.std()


def plot_learning(ax, x, y, cov):
    ax.fill_between(x, y - cov, y + cov)
    ax.plot(x, y)


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
        model = CelloCovarianceEstimationModel()
        model.fit(predictors, covariances)
    elif args.algorithm == 'knn':
        scores = []
        ks = list(range(1,20))
        for k in ks:
            model = KnnModel(k)
            scores.append(covariance_model_performance(model, predictors, covariances))
            print('%.5e var %.5e' % scores[-1])

        scores = np.array(scores)
        fig, ax = plt.subplots()
        plot_learning(ax, ks, scores[:,0], scores[:,1])
        plt.show()

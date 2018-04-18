
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

import recova.util
from recova.util import eprint

def pytorch_is_pd(A):
    try:
        _ = torch.potrf(A)
        return True
    except RuntimeError:
        return False


def pytorch_correlation_repair(C):
    print('Fxing matrix')
    print(C)
    eigvals, eigvecs = torch.eig(C, eigenvectors=True)

    eigvals = torch.max(torch.zeros(6), eigvals[:,0])

    reconstructed = torch.mm(eigvecs, torch.mm(torch.diag(eigvals), eigvecs))

    T = 1. / torch.sqrt(torch.diag(eigvals))
    TT = torch.mm(T, torch.t(T))

    repaired = reconstructed * TT
    return repaired


def pytorch_nearest_pd(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + torch.t(A)) / 2
    _, s, V = torch.svd(B)

    H = torch.mm(torch.t(V), torch.mm(torch.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + torch.t(A2)) / 2

    if pytorch_is_pd(A3):
        return A3

    spacing = np.spacing(torch.norm(A).detach().numpy())

    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.shape[0])
    k = 1
    while not pytorch_is_pd(A3):
        eigvals, _ = torch.eig(A3)
        mineig = torch.min(eigvals)

        A3 += I * (-mineig * k**2 + spacing)
        print(A3)
        print(spacing)
        k += 1

    error = torch.norm(A - A3)
    if error > 1e-6:
        eprint('Warning: high reconstruction error detected when correcting a Positive Definite matrix. {}'.format(error))

    return A3

def kullback_leibler(cov1, cov2):
    """Returns the kullback leibler divergence on a pair of covariances that have the same mean.
    cov1 and cov2 must be numpy matrices.
    See http://bit.ly/2FAYCgu."""

    corrected_cov1 = recova.util.nearestPD(cov1)
    corrected_cov2 = recova.util.nearestPD(cov2)

    det1 = np.linalg.det(corrected_cov1)
    det2 = np.linalg.det(corrected_cov2)

    A = np.trace(np.dot(np.linalg.inv(corrected_cov1), corrected_cov2))
    B = 6.
    C = float(np.log(det1) - np.log(det2))


    kll = 0.5 * (A - B + C)

    return kll

def kullback_leibler_pytorch(cov1, cov2):
    """Returns the kullback leibler divergence on a pair of covariances that have the same mean.
    cov1 and cov2 must be numpy matrices.
    See http://bit.ly/2FAYCgu."""

    corrected_cov1 = nearestPD(cov1.data.numpy())
    corrected_cov2 = nearestPD(cov2.data.numpy())

    det1 = torch.det(corrected_cov1)
    det2 = torch.det(corrected_cov2)

    A = torch.trace(torch.mm(torch.inverse(corrected_cov1), corrected_cov2))
    B = 6.
    C = float(torch.log(det1) - torch.log(det2))

    kll = 0.5 * (A - B + C)

    return kll

def bat_distance(cov1, cov2):
    avg_cov = cov1 + cov2 / 2
    A = torch.det(avg_cov)
    B = torch.det(cov1)
    C = torch.det(cov2)
    return 0.5 * torch.log(A / torch.sqrt(B + C))

def hellinger_distance(cov1, cov2):
    eprint('==== HELLINGER ====')
    eprint('Condition numbers: {} and {}'.format(np.linalg.cond(cov1), np.linalg.cond(cov2)))
    eprint(cov1)
    eprint(np.linalg.det(cov1))
    eprint(cov2)
    eprint(np.linalg.det(cov2))
    return 1.0 - np.power(np.linalg.det(cov1) * np.linalg.det(cov2), 0.25) / np.power(np.linalg.det((cov1 + cov2) / 2.0), 0.5)


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
            # loss_of_i = torch.norm(ys[i] - predictions[i])
            loss_of_i = hellinger_distance(ys[i], predictions[i])
            total_loss += loss_of_i

        eprint('Validation score: {:.2E}'.format(total_loss / len(xs)))
        eprint('Log Validation score: {:.2E}'.format(np.log(total_loss / len(xs))))

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



class MLPModel(CovarianceEstimationModel):
    def __init__(self, configuration=[]):
        pass


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


def size_of_vector(n):
    """The size of a vector representing an nxn upper triangular matrix."""
    return int((n * n + n) / 2.)



class CelloCovarianceEstimationModel(CovarianceEstimationModel):
    def __init__(self, alpha=1e-6):
        self.alpha = alpha

    def fit(self, predictors, covariances):
        self.predictors = Variable(torch.Tensor(predictors))
        self.covariances = Variable(torch.Tensor(covariances))

        # Initialize a distance metric.
        sz_of_vector = size_of_vector(predictors.shape[1])
        self.theta = Variable(torch.randn(sz_of_vector) / 1000., requires_grad=True)

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=100)

        optimizer = optim.SGD([self.theta], lr=1e-6)
        # optimizer = optim.Adam([self.theta], lr=1e-5)
        # optimizer = optim.RMSprop([self.theta], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

        for epoch, (train_set, test_set) in enumerate(selector.split(predictors)):
            optimizer.zero_grad()

            xs_train, xs_validation = Variable(torch.Tensor(predictors[train_set])), Variable(torch.Tensor(predictors[test_set]))
            ys_train, ys_validation = Variable(torch.Tensor(covariances[train_set])), Variable(torch.Tensor(covariances[test_set]))

            sum_of_losses = Variable(torch.Tensor([0.]))
            losses = np.zeros((len(xs_train)))

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
                losses[i] = optimization_loss.data.numpy()
                optimization_loss.backward()
                optimizer.step()

            average_loss = sum_of_losses / len(xs_train)
            median_loss = np.median(np.array(losses))

            validation_score = self.validate(xs_validation.data.numpy(), ys_validation.data.numpy())
            eprint('Avg Optimization Loss: %f' % average_loss)
            eprint('Median optimization loss: %f' % median_loss)
            eprint('Validation score: {:.2E}'.format(validation_score))
            print('{}, {}, {}'.format(epoch, median_loss, validation_score))


    def predict(self, queries):
        torch_queryes = torch.Tensor(queries)
        metric_matrix = self.theta_to_metric_matrix(self.theta)

        predictions = torch.zeros(len(queries),6,6)

        for i, x in enumerate(queries):
            distances = self.compute_distances(self.predictors, metric_matrix, x)
            predictions[i] = self.prediction_from_distances(self.covariances, distances, x)

        return predictions.data.numpy()


    def theta_to_metric_matrix(self, theta):
        up = to_upper_triangular(theta)
        return torch.mm(up, up.transpose(0,1))


    def compute_distances(self, predictors, metric_matrix, predictor):
        pytorch_predictors = torch.Tensor(predictors)
        pytorch_predictor = torch.Tensor(predictor)

        delta = pytorch_predictors - pytorch_predictor.view(1, pytorch_predictor.shape[0])
        lhs = torch.mm(delta, metric_matrix)
        return torch.sum(lhs * delta, 1).squeeze()


    def prediction_from_distances(self, covariances, distances, x):
        zero_distances = distances < 1e-10
        distances.masked_fill(zero_distances, 1.)

        weights = torch.clamp(1. - distances, min=0.)
        predicted_cov = torch.sum(covariances * weights.view(-1,1,1), 0) / torch.sum(weights)

        return predicted_cov




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

    eprint('Loading document')
    input_document = json.load(sys.stdin)
    eprint('Done loading document')

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

        scores = np.array(scores)
        fig, ax = plt.subplots()
        plot_learning(ax, ks, scores[:,0], scores[:,1])
        plt.show()

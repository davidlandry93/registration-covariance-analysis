from math import ceil, sqrt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import sklearn.model_selection

from recova.learning.model import CovarianceEstimationModel
from recova.util import eprint


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


def kullback_leibler_pytorch(cov1, cov2):
    """Returns the kullback leibler divergence on a pair of covariances that have the same mean.
    cov1 and cov2 must be numpy matrices.
    See http://bit.ly/2FAYCgu."""

    det1 = torch.det(cov1)
    det2 = torch.det(cov2)

    A = torch.trace(torch.mm(torch.inverse(cov1), cov2))
    B = 6.
    C = float(torch.log(det1) - torch.log(det2))

    kll = 0.5 * (A - B + C)

    return kll


def pytorch_norm_distance(cov1, cov2):
    return torch.norm(torch.mm(torch.inverse(cov1), cov2) - Variable(torch.eye(6)))

def bat_distance_pytorch(cov1, cov2):
    avg_cov = cov1 + cov2 / 2
    A = torch.det(avg_cov)
    B = torch.det(cov1)
    C = torch.det(cov2)
    return 0.5 * torch.log(A / torch.sqrt(B + C))



class CelloCovarianceEstimationModel(CovarianceEstimationModel):
    def __init__(self, alpha=1e-3, learning_rate=1e-5):
        self.alpha = alpha
        self.learning_rate = learning_rate

    def fit(self, predictors, covariances):
        predictors_train, predictors_test, covariances_train, covariances_test = sklearn.model_selection.train_test_split(predictors, covariances, test_size=0.25)


        self.predictors = Variable(torch.Tensor(predictors_train))
        self.covariances = Variable(torch.Tensor(covariances_train))

        # Initialize a distance metric.
        sz_of_vector = size_of_vector(predictors.shape[1])
        self.theta = Variable(torch.randn(sz_of_vector) / 1000., requires_grad=True)

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=20)

        optimizer = optim.SGD([self.theta], lr=self.learning_rate)

        validation_losses = []
        optimization_losses = []
        for epoch, (train_set, test_set)  in enumerate(selector.split(predictors_train)):
            optimizer.zero_grad()

            xs_train, xs_validation = Variable(torch.Tensor(predictors_train[train_set])), Variable(torch.Tensor(predictors_train[test_set]))
            ys_train, ys_validation = Variable(torch.Tensor(covariances_train[train_set])), Variable(torch.Tensor(covariances_train[test_set]))

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

            try:
                validation_score = self.validate(predictors_test, covariances_test)
            except ValueError:
                raise

            eprint('Avg Optimization Loss: %f' % average_loss)
            eprint('Median optimization loss: %f' % median_loss)
            eprint('Validation score: {:.2E}'.format(validation_score))
            eprint('Epoch: %d' % epoch)

            validation_losses.append(validation_score)
            optimization_losses.append(average_loss)

        return {
            'what': 'model learning',
            'metadata': self.metadata(),
            'validation_loss': validation_losses,
            'optimization_loss': optimization_losses,
            'model': self.theta.tolist()
        }


    def metadata(self):
        return {
            'algorithm': 'cello',
            'learning_rate': self.learning_rate,
            'alpha': self.alpha
        }


    def predict(self, queries):
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
        # zero_distances = distances < 1e-10
        # distances.masked_fill(zero_distances, 1.)
        # weights = torch.clamp(1. - distances, min=0.)

        weights = torch.exp(-distances)

        predicted_cov = torch.sum(covariances * weights.view(-1,1,1), 0) / torch.sum(weights)

        # eprint('Sum of weights: %f' % torch.sum(weights))
        return predicted_cov

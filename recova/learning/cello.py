
import json
from math import ceil, sqrt
import numpy as np
import time
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


def size_of_triangular_vector(n):
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
    def __init__(self, alpha=1e-4, learning_rate=1e-5, n_iterations=100, beta=1000., train_set_size=0.3, convergence_window=20):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = beta
        self.train_set_size = train_set_size
        self.convergence_window = convergence_window

    def fit(self, predictors, covariances, train_set=None, test_set=None):
        eprint('Training with descriptors of size {}'.format(predictors.shape[1]))

        if train_set and test_set:
            training_indices = train_set
            test_indices = test_set
        else:
            training_indices, test_indices = sklearn.model_selection.train_test_split(list(range(len(predictors))), test_size=0.3)

        predictors_train, predictors_test, covariances_train, covariances_test = predictors[training_indices], predictors[test_indices], covariances[training_indices], covariances[test_indices]

        self.model_predictors = Variable(torch.Tensor(predictors_train))
        self.model_covariances = Variable(torch.Tensor(covariances_train))

        predictors_validation = torch.Tensor(predictors_test)
        covariances_validation = torch.Tensor(covariances_test)

        result = self._fit(predictors_validation, covariances_validation)

        result['train_set'] = training_indices
        result['validation_set'] = test_indices

        return result



    def _fit(self, predictors_validation, covariances_validation):
        """
        Given a validation set, train weights theta that optimize the validation error of the model.
        """
        self.theta = self.create_metric_weights()

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10)

        # optimizer = optim.SGD([self.theta], lr=self.learning_rate)
        optimizer = optim.Adam([self.theta], lr=self.learning_rate)

        validation_losses = []
        validation_stds = []
        optimization_losses = []
        optimization_stds = []
        validation_errors_log = []
        optimization_errors_log = []

        kll_errors_log = []
        kll_validation_losses = []
        kll_validation_stds = []

        epoch = 0
        keep_going = True

        best_loss = np.inf
        best_model = []
        n_epoch_without_improvement = 0

        while (epoch < self.n_iterations or self.n_iterations == 0) and keep_going and n_epoch_without_improvement < self.convergence_window:
            epoch_start = time.time()
            optimizer.zero_grad()

            losses = Variable(torch.zeros(len(self.model_predictors)))

            for i in range(len(self.model_predictors)):
                # eprint(self.theta)
                metric_matrix = self.theta_to_metric_matrix(self.theta)
                # eprint(metric_matrix)

                distances = self._compute_distances(self.model_predictors, metric_matrix, self.model_predictors[i])

                # eprint(self.distances_to_weights(distances))

                prediction = self.prediction_from_distances(self.model_covariances, distances)

                loss_A = torch.log(torch.norm(prediction))
                # loss_B = torch.norm(torch.mm(torch.inverse(prediction), self.model_covariances[i]) - identity)

                loss_B = torch.trace(torch.mm(torch.inverse(prediction), self.model_covariances[i]))

                nonzero_distances = torch.gather(distances, 0, torch.nonzero(distances).squeeze())
                regularization_term = torch.sum(torch.log(nonzero_distances))

                optimization_loss = (1 - self.alpha) * (loss_A + loss_B) + self.alpha * regularization_term
                losses[i] = optimization_loss

                # eprint('A: {}, B: {}, C: {}'.format(loss_A, loss_B, regularization_term))

                optimization_loss.backward()
                optimizer.step()

            predictions = self._predict(self.model_predictors)
            indiv_optimization_errors = self._validation_errors(predictions, self.model_covariances)
            optimization_score = torch.mean(indiv_optimization_errors).data.numpy().item()
            optimization_errors_log.append(indiv_optimization_errors.data.numpy().tolist())
            optimization_losses.append(optimization_score)
            optimization_stds.append(torch.std(indiv_optimization_errors).data.numpy().item())

            metric_matrix = self.theta_to_metric_matrix(self.theta)

            if self.queries_have_neighbor(self.model_predictors, metric_matrix, Variable(predictors_validation)):
                predictions = Variable(self._predict(predictors_validation))
                validation_errors = self._validation_errors(predictions, Variable(covariances_validation)).data
                validation_score = torch.mean(validation_errors)

                klls = self._kll(predictions, covariances_validation)
                kll_errors_log.append(klls.data.numpy().tolist())
                kll_validation_losses.append(torch.mean(klls).data.numpy().item())
                kll_validation_stds.append(torch.std(klls).data.numpy().item())


                eprint('-- Validation of epoch %d --' % epoch)

                if validation_score < best_loss:
                    eprint('** New best model! **')
                    n_epoch_without_improvement = 0
                    best_loss = validation_score
                    best_model = self.export_model()
                else:
                    n_epoch_without_improvement += 1

                eprint('Avg Optim Loss:     {:.8E}'.format(optimization_score))
                eprint('Validation score:   {:.8E}'.format(validation_score))
                eprint('Validation std:     {:.8E}'.format(validation_errors.std()))
                eprint('Validation kll:     {:.8E}'.format(klls.mean()))
                eprint('Validation kll std: {:.8E}'.format(klls.std()))
                eprint('N epoch without improvement: %d' % n_epoch_without_improvement)
                eprint()

                validation_errors_log.append(validation_errors.numpy().tolist())
                validation_losses.append(validation_score.numpy().item())
                validation_stds.append(torch.std(validation_errors).numpy().item())
            else:
                keep_going = False
                eprint('Stopping because elements in the validation dataset have no neighbors.')

            eprint('Epoch took {} seconds'.format(time.time() - epoch_start))
            epoch = epoch + 1


        return {
            'best_loss': float(best_loss),
            'metadata': self.metadata(),
            'model': best_model,
            'optimization_errors': optimization_errors_log,
            'optimization_loss': optimization_losses,
            'optimization_std': optimization_stds,
            'validation_errors': validation_errors_log,
            'validation_loss': validation_losses,
            'validation_std': validation_stds,
            'kll_validation': kll_validation_losses,
            'kll_std': kll_validation_stds,
            'kll_errors': kll_errors_log,
            'what': 'model learning',
        }

    def create_metric_weights(self):
        size_of_vector = size_of_triangular_vector(self.model_predictors.shape[1])
        return Variable(torch.randn(size_of_vector) / self.beta, requires_grad=True)

    def validation_errors(self, xs, ys):
        predictions = self._predict(torch.Tensor(xs))
        return self._validation_errors(predictions, torch.Tensor(ys))

    def _validation_errors(self, ys_predicted, ys_validation):
        losses = (ys_predicted - ys_validation).pow(2.0)
        losses = torch.sqrt(losses.sum(dim=2).sum(dim=1))

        return losses

    def validate(self, xs, ys):
        return self._validate(torch.Tensor(xs), torch.Tensor(xs))

    def _validate(self, xs, ys):
        validation_errors = self._validation_errors(xs,ys)
        return torch.mean(validation_errors)

    def _kll(self, ys_predicted, ys_validation):
        kll_errors = torch.zeros(len(ys_predicted))

        for i in range(len(ys_predicted)):
            kll_errors[i] = kullback_leibler_pytorch(ys_predicted[i], ys_validation[i]) + kullback_leibler_pytorch(ys_validation[i], ys_predicted[i]) / 2.

        return kll_errors


    def metadata(self):
        return {
            'algorithm': 'cello',
            'learning_rate': self.learning_rate,
            'alpha': self.alpha,
            'logging_rate': 1,
        }


    def predict(self, queries):
        return self._predict(torch.Tensor(queries)).numpy()

    def _predict(self, predictors):
        metric_matrix = self.theta_to_metric_matrix(self.theta)
        predictions = Variable(torch.zeros(len(predictors),6,6))

        for i in range(len(predictors)):
            distances = self._compute_distances(self.model_predictors, metric_matrix, predictors[i])
            predictions[i] = self.prediction_from_distances(self.model_covariances, distances)

        return predictions



    def theta_to_metric_matrix(self, theta):
        up = to_upper_triangular(theta)
        return torch.mm(up, up.transpose(0,1))

    def _metric_matrix(self):
        return self.theta_to_metric_matrix(self.theta)

    def metric_matrix(self):
        return self._metric_matrix().numpy()


    def _compute_distances(self, predictors, metric_matrix, predictor):
        delta = predictors - predictor.view(1, predictor.shape[0])
        lhs = torch.mm(delta, metric_matrix)
        return torch.sum(lhs * delta, 1).squeeze()

    def compute_distances(self, predictor):
        metric_matrix = self.theta_to_metric_matrix(self.theta)
        return self._compute_distances(self.model_predictors, metric_matrix, torch.Tensor(predictor)).detach().numpy()

    def distances_to_weights(self, distances):
        # zero_distances = distances < 1e-10
        # distances.masked_fill(zero_distances, 1.)
        # # eprint(distances)
        # weights = torch.clamp(1. - distances, min=0.)

        weights = torch.exp(-distances)

        # Points that were a perfect match should have no weight.
        weights.masked_fill(weights == 1.0, 0.0)
        return weights


    def prediction_from_distances(self, covariances, distances):
        weights = self.distances_to_weights(distances)

        predicted_cov = torch.sum(covariances * weights.view(-1,1,1), 0)
        predicted_cov /= torch.sum(weights)

        # eprint('Sum of weights: %f' % torch.sum(weights))
        return predicted_cov

    def queries_have_neighbor(self, predictors, metric_matrix, examples):
        """Check if every descriptor in queries has at least one neighbour in the learned model."""

        n_predictors = len(predictors)
        distances = Variable(torch.zeros([len(examples), n_predictors]))
        for i in range(n_predictors):
            distances[:, i] = self._compute_distances(examples, metric_matrix, predictors[i])

        weights = self.distances_to_weights(distances)
        sum_of_weights = torch.sum(weights, dim=0)

        return not (sum_of_weights == 0.0).any()

    def export_model(self):
       return {
           'theta': self.theta.data.numpy().tolist(),
           'covariances' : self.model_covariances.data.numpy().tolist(),
           'predictors': self.model_predictors.data.numpy().tolist()
       }

    def save_model(self, location):
       with open(location, 'w') as f:
           json.dump(self.export_model(), f)

    def load_model(self, location):
        with open(location) as f:
            self.import_model(json.load(f))

    def import_model(self, model):
        self.theta = torch.Tensor(model['theta'])
        self.model_covariances = torch.Tensor(model['covariances'])
        self.model_predictors = torch.Tensor(model['predictors'])



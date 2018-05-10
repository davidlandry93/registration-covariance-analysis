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
    def __init__(self, alpha=1e-4, learning_rate=1e-5, n_iterations=100, beta=1000., train_set_size=0.3):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = beta
        self.train_set_size = train_set_size

    def fit(self, predictors, covariances):
        training_indices, test_indices = sklearn.model_selection.train_test_split(list(range(len(predictors))), test_size=0.3)

        predictors_train, predictors_test, covariances_train, covariances_test = predictors[training_indices], predictors[test_indices], covariances[training_indices], covariances[test_indices]

        self.model_predictors = torch.Tensor(predictors_train).cuda()
        self.model_covariances = torch.Tensor(covariances_train).cuda()

        predictors_validation = torch.Tensor(predictors_test).cuda()
        covariances_validation = torch.Tensor(covariances_test).cuda()

        result = self._fit(predictors_validation, covariances_validation)



    def _fit(self, predictors_validation, covariances_validation):
        """
        Given a validation set, train weights theta that optimize the validation error of the model.
        """
        self.theta = self.create_metric_weights()

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10)

        optimizer = optim.SGD([self.theta], lr=self.learning_rate)

        validation_losses = []
        validation_stds = []
        optimization_losses = []
        validation_errors_log = []
        optimization_errors = []

        epoch = 0
        keep_going = True

        best_loss = np.inf
        best_model = []
        n_epoch_without_improvement = 0

        identity = torch.eye(6).cuda()

        while epoch < self.n_iterations and keep_going and n_epoch_without_improvement < 20:
            epoch_start = time.time()
            optimizer.zero_grad()

            losses = torch.zeros(len(self.model_predictors)).cuda()

            for i in range(len(self.model_predictors)):
                metric_matrix = self.theta_to_metric_matrix(self.theta)

                distances = self.compute_distances(self.model_predictors, metric_matrix, self.model_predictors[i])
                prediction = self.prediction_from_distances(self.model_covariances, distances)

                loss_A = torch.log(torch.norm(prediction))
                loss_B = torch.log(torch.norm(torch.mm(torch.inverse(prediction), self.model_covariances[i]) - identity))
                nonzero_distances = torch.gather(distances, 0, torch.nonzero(distances).squeeze())
                regularization_term = torch.sum(torch.log(nonzero_distances))

                optimization_loss = (1 - self.alpha) * (loss_A + loss_B) + self.alpha * regularization_term
                losses[i] = optimization_loss

                optimization_loss.backward()
                optimizer.step()

            indiv_optimization_errors = self._validation_errors(self.model_predictors, self.model_covariances)
            optimization_errors.append(indiv_optimization_errors.tolist())


            metric_matrix = self.theta_to_metric_matrix(self.theta)

            if self.queries_have_neighbor(self.model_predictors, metric_matrix, predictors_validation):
                validation_errors = self._validation_errors(predictors_validation, covariances_validation).data
                validation_score = torch.mean(validation_errors).data
                optimization_score = torch.mean(indiv_optimization_errors).data

                eprint('-- Validation of epoch %d --' % epoch)

                if validation_score < best_loss:
                    eprint('** New best model! **')
                    n_epoch_without_improvement = 0
                    best_loss = validation_score
                    best_model = self.theta.detach().cpu().numpy()
                else:
                    n_epoch_without_improvement += 1

                eprint('Avg Optim Loss:   {:.4E}'.format(optimization_score))
                eprint('Validation score: {:.4E}'.format(validation_score))
                eprint('N epoch without improvement: %d' % n_epoch_without_improvement)
                eprint()

                validation_errors_log.append(validation_errors.cpu().detach().numpy().tolist())
                validation_losses.append(validation_score)
                optimization_losses.append(optimization_score)
            else:
                keep_going = False
                eprint('Stopping because elements in the validation dataset have no neighbors.')

            epoch = epoch + 1
            eprint('Epoch took {} seconds'.format(time.time() - epoch_start))


        return {
            'what': 'model learning',
            'metadata': self.metadata(),
            'train_set': training_indices,
            'validation_set': test_indices,
            'validation_loss': validation_losses,
            'validation_std': validation_stds,
            'validation_errors': validation_errors,
            'optimization_loss': optimization_losses,
            'optimization_errors': optimization_errors,
            'model': best_model.tolist()
        }

    def create_metric_weights(self):
        size_of_vector = size_of_triangular_vector(self.model_predictors.shape[1])
        return Variable(torch.randn(size_of_vector).cuda() / self.beta, requires_grad=True)

    def validation_errors(self, xs, ys):
        return self._validation_errors(torch.Tensor(xs).cuda(), torch.Tensor(ys).cuda())

    def _validation_errors(self, xs, ys):
        predictions = self._predict(xs)

        losses = (predictions - ys).pow(2.0)
        losses = torch.sqrt(losses.sum(dim=2).sum(dim=1))


        return losses

    def validate(self, xs, ys):
        return self._validate(torch.Tensor(xs).cuda(), torch.Tensor(xs).cuda())

    def _validate(self, xs, ys):
        validation_errors = self._validation_errors(xs,ys)
        return torch.mean(validation_errors)



    def metadata(self):
        return {
            'algorithm': 'cello',
            'learning_rate': self.learning_rate,
            'alpha': self.alpha
        }


    def predict(self, queries):
        return self._predict(torch.Tensor(queries).cuda())

    def _predict(self, predictors):
        metric_matrix = self.theta_to_metric_matrix(self.theta)
        predictions = torch.zeros(len(predictors),6,6).cuda()

        for i in range(len(predictors)):
            distances = self.compute_distances(self.model_predictors, metric_matrix, predictors[i])
            predictions[i] = self.prediction_from_distances(self.model_covariances, distances)

        return predictions



    def theta_to_metric_matrix(self, theta):
        cpu_theta = theta.cpu()
        up = to_upper_triangular(cpu_theta)
        return torch.mm(up, up.transpose(0,1)).cuda()


    def compute_distances(self, predictors, metric_matrix, predictor):
        pytorch_predictors = predictors
        pytorch_predictor = predictor

        delta = pytorch_predictors.cuda() - pytorch_predictor.view(1, pytorch_predictor.shape[0]).cuda()
        lhs = torch.mm(delta, metric_matrix)
        return torch.sum(lhs * delta, 1).squeeze()

    def distances_to_weights(self, distances):
        # zero_distances = distances < 1e-10
        # distances.masked_fill(zero_distances, 1.)
        # eprint(distances)
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
        distances = torch.zeros([len(examples), n_predictors])
        for i in range(n_predictors):
            distances[:, i] = self.compute_distances(examples, metric_matrix, predictors[i])

        weights = self.distances_to_weights(distances)
        sum_of_weights = torch.sum(weights, dim=0)

        return not (sum_of_weights == 0.0).any()


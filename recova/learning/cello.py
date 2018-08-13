
import json
import gc
import logging
from math import ceil, sqrt
import numpy as np
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import sklearn.model_selection
import sys

from recova.learning.model import CovarianceEstimationModel
from recova.learning.preprocessing import preprocessing_factory
from recova.util import eprint, kullback_leibler, wishart_kl_divergence


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


def kullback_leibler_pytorch(base, base_df, prediction, prediction_df):
    """Returns the kullback leibler divergence on a pair of covariances that have the same mean.
    cov1 and cov2 must be numpy matrices.
    See http://bit.ly/2FAYCgu. """

    # In pytorch 0.4.0 det will allow us to compute the kll directly in torch,
    # in the meantime we transfer the computation to numpy.

    base_np, prediction_np = base.numpy(), prediction.numpy()
    return wishart_kl_divergence(base_np, base_df, prediction_np, prediction_df)


def pytorch_norm_distance(cov1, cov2):
    return torch.norm(torch.mm(torch.inverse(cov1), cov2) - Variable(torch.eye(6)))

def bat_distance_pytorch(cov1, cov2):
    avg_cov = cov1 + cov2 / 2
    A = torch.det(avg_cov)
    B = torch.det(cov1)
    C = torch.det(cov2)
    return 0.5 * torch.log(A / torch.sqrt(B + C))

def identity_preprocessing(ys):
    return (ys)

def identity_postprocessing(ys):
    return (ys)


class CelloCovarianceEstimationModel(CovarianceEstimationModel):
    def __init__(self, alpha=1e-4, learning_rate=1e-5, n_iterations=100, beta=1000., train_set_size=0.3, patience=20, preprocessing='identity', min_delta = 1e-4):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = beta
        self.train_set_size = train_set_size
        self.patience = patience
        self.preprocessing = preprocessing
        self.min_delta = 1e-4
        self.logger = logging.getLogger()

    def fit(self, predictors, covariances, train_set=None, test_set=None):
        self.logger.info('Training with descriptors of size {}'.format(predictors.shape[1]))

        if train_set and test_set:
            training_indices = train_set
            test_indices = test_set
        else:
            training_indices, test_indices = sklearn.model_selection.train_test_split(list(range(len(predictors))), test_size=0.3)

        preprocessed_covariances = self.preprocessing.process(covariances)

        predictors_train, predictors_test, covariances_train, covariances_test = predictors[training_indices], predictors[test_indices], preprocessed_covariances[training_indices], preprocessed_covariances[test_indices]

        self.model_predictors = Variable(torch.Tensor(predictors_train))
        self.model_predictors_cuda = self.model_predictors.cuda()
        self.model_covariances = Variable(torch.Tensor(covariances_train))
        self.model_covariances_cuda = self.model_covariances.cuda()

        predictors_validation = Variable(torch.Tensor(predictors_test))
        covariances_validation = Variable(torch.Tensor(covariances_test))

        print('Size of predictors train: %d' % (sys.getsizeof(predictors_train) / 1024))
        print('Size of covariances train: %d' % (sys.getsizeof(covariances_train) / 1024))
        print('Size of predictors validation: %d' % (sys.getsizeof(predictors_test) / 1024))
        print('Size of covariances validation: %d' % (sys.getsizeof(covariances_train) / 1024))

        result = self._fit(predictors_validation, covariances_validation)

        result['train_set'] = training_indices
        result['validation_set'] = test_indices

        return result



    def _fit(self, predictors_validation, covariances_validation):
        """
        Given a validation set, train weights theta that optimize the validation error of the model.
        """

        predictors_validation_cuda = predictors_validation.cuda()

        self.theta = self.create_metric_weights()

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10)

        optimizer = optim.SGD([self.theta], lr=self.learning_rate)
        # optimizer = optim.Adam([self.theta], lr=self.learning_rate)

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
        n_epoch_without_min_delta = 0

        while (epoch < self.n_iterations or self.n_iterations == 0) and keep_going and n_epoch_without_improvement < self.patience and n_epoch_without_min_delta < self.patience:
            self.logger.debug('Starting epoch %d' % epoch)

            epoch_start = time.time()
            optimizer.zero_grad()

            losses = Variable(torch.zeros(len(self.model_predictors)))
            optimization_loss = 0.0
            metric_matrix = self.theta_to_metric_matrix(self.theta)

            perms = torch.randperm(len(self.model_predictors))
            for i in perms:
                distances = self._compute_distances_cuda(self.model_predictors_cuda, metric_matrix.cuda(), self.model_predictors[i].cuda())
                prediction = self._prediction_from_distances_cuda(self.model_covariances_cuda, distances).cpu()
                # prediction = self._prediction_from_distances_cu(self.model_covariances, distances)


                det_pred = torch.det(prediction)
                log_det = torch.log(det_pred + 1e-15)

                loss_A = log_det
                # loss_B = torch.norm(torch.mm(torch.inverse(prediction), self.model_covariances[i]) - identity)

                loss_B = torch.trace(torch.mm(torch.inverse(prediction), self.model_covariances[i]))

                distances_cpu = distances.cpu()
                nonzero_distances = torch.gather(distances_cpu, 0, torch.nonzero(distances_cpu).squeeze())
                regularization_term = torch.sum(torch.log(nonzero_distances))

                loss_of_pair = (1 - self.alpha) * (loss_A + loss_B) + self.alpha * regularization_term

                optimization_loss += loss_of_pair
                losses[i] = loss_of_pair

                if i % 1 == 0:
                    self.logger.debug('Backprop')
                    optimization_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    optimization_loss = 0.0
                    metric_matrix = self.theta_to_metric_matrix(self.theta)


            predictions = self._predict(self.model_predictors_cuda)
            self.logger.debug('Predictions have size %d kb' % sys.getsizeof(predictions))

            indiv_optimization_errors = self._validation_errors(predictions, self.model_covariances.data)
            self.logger.debug('Indiv optimization errors have size %d kb' % sys.getsizeof(indiv_optimization_errors))

            optimization_score = torch.mean(indiv_optimization_errors)
            optimization_errors_log.append(indiv_optimization_errors.numpy().tolist())
            optimization_losses.append(optimization_score.item())
            optimization_stds.append(torch.std(indiv_optimization_errors).item())

            metric_matrix = self.theta_to_metric_matrix(self.theta)

            if self.queries_have_neighbor_cuda(self.model_predictors_cuda, metric_matrix.cuda(), predictors_validation_cuda):
                predictions = self._predict(predictors_validation_cuda)
                validation_errors = self._validation_errors(predictions, covariances_validation.data)
                validation_score = torch.mean(validation_errors)

                klls = self._kll(predictions, covariances_validation.data)
                kll_errors_log.append(klls.numpy().tolist())
                kll_validation_losses.append(torch.mean(klls).numpy().item())
                kll_validation_stds.append(torch.std(klls).numpy().item())


                eprint('-- Validation of epoch %d --' % epoch)
                if validation_score < best_loss:
                    eprint('** New best model! **')
                    n_epoch_without_improvement = 0
                    best_loss = validation_score
                    best_model = self.export_model()
                else:
                    n_epoch_without_improvement += 1

                eprint('Avg Optim Loss:     {:.5E}'.format(optimization_score))
                eprint('Validation score:   {:.5E}'.format(validation_score))
                eprint()
                if epoch > 0:
                    eprint('Optim. delta:       {:.5E}'.format(optimization_losses[-2] - optimization_losses[-1]))
                    eprint('Validation delta:   {:.5E}'.format(validation_losses[-1] - validation_score))
                    eprint()
                eprint('Validation std:     {:.5E}'.format(validation_errors.std()))
                eprint('Validation kll:     {:.5E}'.format(klls.mean()))
                eprint('Validation kll std: {:.5E}'.format(klls.std()))
                eprint('N epoch without improvement: %d' % n_epoch_without_improvement)
                eprint('N epoch without min delta:   %d' % n_epoch_without_min_delta)
                eprint()

                validation_errors_log.append(validation_errors.numpy().tolist())
                validation_losses.append(validation_score.numpy().tolist())
                validation_stds.append(torch.std(validation_errors).numpy().tolist())
            else:
                keep_going = False
                eprint('Stopping because elements in the validation dataset have no neighbors.')

            if (epoch > 0) and (validation_losses[-1] - validation_losses[-2] > -1.0 * self.min_delta):
                n_epoch_without_min_delta += 1
            else:
                n_epoch_without_min_delta = 0


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

    # def _validation_errors(self, ys_predicted, ys_validation):
    #     losses = torch.abs(ys_predicted - ys_validation)
    #     losses = losses.sum(dim=2).sum(dim=1)

    #     return losses

    def validate(self, xs, ys):
        return self._validate(torch.Tensor(xs), torch.Tensor(xs))

    def _validate(self, xs, ys):
        validation_errors = self._validation_errors(xs,ys)
        return torch.mean(validation_errors)

    def _kll(self, ys_predicted, ys_validation):
        kll_errors = torch.zeros(len(ys_predicted))

        for i in range(len(ys_predicted)):
            kll_errors[i] = kullback_leibler_pytorch(ys_validation[i] / 6, 6, ys_predicted[i] / 100, 100)

        return kll_errors


    def metadata(self):
        return {
            'algorithm': 'cello',
            'learning_rate': self.learning_rate,
            'alpha': self.alpha,
            'logging_rate': 1,
            'min_delta': self.min_delta,
            'preprocessing': repr(self.preprocessing),
            'patience': self.patience
        }


    def predict(self, queries):
        predictions = self._predict(Variable(torch.Tensor(queries))).data.numpy()
        post_processed = self.preprocessing.unprocess(predictions)

        return post_processed

    def _predict(self, predictors):
        metric_matrix = self.theta_to_metric_matrix(self.theta)
        metric_matrix_cuda = metric_matrix.cuda()

        self.logger.debug('Preparing to predict %d values' % len(predictors))
        eprint(self.model_covariances.shape[1])
        eprint(self.model_covariances.shape[2])
        predictions = torch.zeros(len(predictors),self.model_covariances.shape[1], self.model_covariances.shape[2])

        for i in range(len(predictors)):
            self.logger.debug('Predicting value for predictor %d ' % i)
            distances = self._compute_distances_cuda(self.model_predictors_cuda, metric_matrix_cuda, predictors[i].cuda())

            predictions[i] = self._prediction_from_distances_cuda(self.model_covariances_cuda, distances).data

        return predictions



    def theta_to_metric_matrix(self, theta):
        up = to_upper_triangular(theta)

        up_cuda = up.cuda()

        return torch.mm(up_cuda, up_cuda.transpose(0,1)).cpu()

    def _metric_matrix(self):
        return self.theta_to_metric_matrix(self.theta)

    def metric_matrix(self):
        return self._metric_matrix().numpy()


    def _compute_distances(self, predictors, metric_matrix, predictor):
        cuda_metric_matrix = metric_matrix.cuda()
        cuda_predictors = predictors.cuda()
        cuda_predictor = predictor.cuda()

        return self._compute_distances_cuda(cuda_predictors, cuda_metric_matrix, cuda_predictor).cpu()

    def _compute_distances_cuda(self, predictors, metric_matrix, predictor):
        delta = predictors - predictor.view(1, predictor.shape[0])
        lhs = torch.mm(delta, metric_matrix)
        return torch.sum(lhs * delta, 1).squeeze()


    def compute_distances(self, predictor):
        metric_matrix = self.theta_to_metric_matrix(self.theta)
        return self._compute_distances(self.model_predictors, metric_matrix, Variable(torch.Tensor(predictor))).data.numpy()

    def distances_to_weights(self, distances):
        zero_distances = distances < 1e-10
        # eprint(zero_distances.sum())
        distances.masked_fill_(zero_distances, 1000000.)
        # distances[zero_distances] = 1000000.
        # weights = torch.clamp(1. - distances, min=0.)

        # srt, _ = distances.sort()
        # eprint('dsts')
        # eprint(srt)

        weights = torch.exp(-distances)
        # weights = 1.0 / distances

        # srt, _ = weights.sort()
        # eprint('weights')
        # eprint(srt)

        # Points that were a perfect match should have no weight.
        # weights.masked_fill(weights == 1.0, 0.0)
        return weights


    def prediction_from_distances(self, covariances, distances):
        weights = self.distances_to_weights(distances)
        sum_of_weights = weights.sum()
        reshaped_weights = weights.view(-1, 1, 1)

        weighted_cov = covariances * reshaped_weights

        predicted_cov = torch.sum(weighted_cov, 0)
        predicted_cov /= sum_of_weights

        return predicted_cov


    def prediction_from_distances_eigen(self, covariances, distances):
        sum_eigvals = np.zeros(6)
        sum_eigvecs = np.zeros((6,6))
        for covariance in covariances:
            cov_numpy = covariance.numpy()
            eigvals, eigvecs = np.linalg.eig(cov_numpy)

            sum_eigvals += eigvals
            sum_eigvecs += eigvecs

        q = sum_eigvecs / len(covariances)
        dia = np.diag(sum_eigvals) / len(covariances)

        prediction = np.dot(q, np.dot(dia, q.T))

        return torch.Tensor(prediction)


    def _prediction_from_distances_cuda(self, covariances, distances):
        weights = self.distances_to_weights(distances)

        predicted_cov = torch.sum(covariances * weights.view(-1,1,1), 0)
        predicted_cov /= torch.sum(weights)

        # eprint('Sum of weights: %f' % torch.sum(weights))
        return predicted_cov

    def queries_have_neighbor(self, predictors, metric_matrix, examples):
        """Check if every descriptor in queries has at least one neighbour in the learned model."""
        self.logger.debug('Calling queries have neighbors...')

        n_predictors = len(predictors)
        self.logger.debug('Computing distances for %d examples and %d predictors' % (len(examples), n_predictors))
        distances = torch.zeros([len(examples), n_predictors])

        for i, q in enumerate(predictors):
            distances = self._compute_distances(examples, metric_matrix, q)
            weights = self.distances_to_weights(distances)
            if torch.sum(weights).data[0] == 0.:
                self.logger.info('Predictor %d does not have a neighbor' % i)
                return False

        self.logger.debug('Done calling queries have neighbors.')
        return True


    def queries_have_neighbor_cuda(self, predictors, metric_matrix, examples):
        n_predictors = len(predictors)
        self.logger.debug('Computing distances for %d examples and %d predictors' % (len(examples), n_predictors))
        distances = torch.zeros([len(examples), n_predictors])

        for i, q in enumerate(predictors):
            distances = self._compute_distances_cuda(examples, metric_matrix, q)
            weights = self.distances_to_weights(distances)
            if torch.sum(weights).data[0] == 0.:
                self.logger.info('Predictor %d does not have a neighbor' % i)
                return False

        self.logger.debug('Done calling queries have neighbors.')
        return True



    def export_model(self):
       return {
           'preprocessing': self.preprocessing.export(),
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
        self.theta = Variable(torch.Tensor(model['theta']))
        self.model_covariances = Variable(torch.Tensor(model['covariances']))
        self.model_covariances_cuda = self.model_covariances.cuda()
        self.model_predictors = Variable(torch.Tensor(model['predictors']))
        self.model_predictors_cuda = self.model_predictors.cuda()

        preprocessing_algo = preprocessing_factory(model['preprocessing']['name'])
        preprocessing_algo.import_model(model['preprocessing'])
        self.preprocessing = preprocessing_algo


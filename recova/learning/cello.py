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
    def __init__(self, alpha=1e-4, learning_rate=1e-5, n_iterations=100, beta=1000.):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = beta

    def fit(self, predictors, covariances):
        training_indices, test_indices = sklearn.model_selection.train_test_split(list(range(len(predictors))), test_size=0.3)
        predictors_train, predictors_test, covariances_train, covariances_test = predictors[training_indices], predictors[test_indices], covariances[training_indices], covariances[test_indices]

        predictors_test = torch.Tensor(predictors_test, device='cuda')
        covariances_test = torch.Tensor(covariances_test, device='cuda')
        covariances_test = covariances_test.cuda()


        self.predictors = Variable(torch.Tensor(predictors_train, device='cuda'))
        self.covariances = Variable(torch.Tensor(covariances_train, device='cuda'))
        self.covariances = self.covariances.cuda()

        # Initialize a distance metric.
        sz_of_vector = size_of_vector(predictors.shape[1])
        self.theta = Variable(torch.randn(sz_of_vector, device='cuda') / self.beta, requires_grad=True)

        selector = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10)

        optimizer = optim.SGD([self.theta], lr=self.learning_rate)

        validation_losses = []
        validation_stds = []
        optimization_losses = []
        validation_errors = []
        optimization_errors = []

        epoch = 0
        keep_going = True

        best_loss = np.inf
        best_model = []
        n_epoch_without_improvement = 0

        while epoch < self.n_iterations and keep_going and n_epoch_without_improvement < 20:
            optimizer.zero_grad()
            train_set, test_set = next(selector.split(predictors_train))

            xs_train, xs_validation = Variable(torch.Tensor(predictors_train[train_set], device='cuda')), Variable(torch.Tensor(predictors_train[test_set]))
            ys_train, ys_validation = Variable(torch.Tensor(covariances_train[train_set])), Variable(torch.Tensor(covariances_train[test_set]))
            ys_train = ys_train.cuda()

            sum_of_losses = Variable(torch.Tensor([0.], device='cuda'))
            losses = np.zeros((len(xs_train)))



            for i, x in enumerate(xs_train):
                metric_matrix = self.theta_to_metric_matrix(self.theta)

                distances = self.compute_distances(xs_train, metric_matrix, x)

                prediction = self.prediction_from_distances(ys_train, distances)

                loss_A = torch.log(torch.norm(prediction))
                loss_B = torch.log(torch.norm(torch.mm(torch.inverse(prediction), ys_train[i]) - Variable(torch.eye(6, device='cuda'))))
                nonzero_distances = torch.gather(distances, 0, torch.nonzero(distances).squeeze())
                regularization_term = torch.sum(torch.log(nonzero_distances))

                optimization_loss = (1 - self.alpha) * (loss_A + loss_B) + self.alpha * regularization_term
                optimization_loss = optimization_loss.cpu()
                sum_of_losses += optimization_loss
                losses[i] = optimization_loss.data.numpy()


                optimization_loss.backward()
                optimizer.step()

            indiv_optimization_errors = self.validation_errors(xs_train, ys_train)
            optimization_errors.append(indiv_optimization_errors.tolist())


            average_loss = sum_of_losses / len(xs_train)
            median_loss = np.median(np.array(losses))

            metric_matrix = self.theta_to_metric_matrix(self.theta)
            queries_have_neighbor = self.queries_have_neighbor(predictors_test, metric_matrix, self.predictors)

            epoch = epoch + 1

            if queries_have_neighbor:
                validation_score, validation_std = self.validate(predictors_test, covariances_test)

                eprint('-- Validation of epoch %d --' % epoch)

                if validation_score < best_loss:
                    eprint('** New best model! **')
                    n_epoch_without_improvement = 0
                    best_loss = validation_score
                    best_model = self.theta.detach().cpu().numpy()
                else:
                    n_epoch_without_improvement += 1

                eprint('Avg Optim Loss:   {:.4E}'.format(average_loss.data[0]))
                eprint('Validation score: {:.4E}'.format(validation_score))
                eprint('Validation std:   {:.4E}'.format(validation_std))
                eprint('N epoch without improvement: %d' % n_epoch_without_improvement)
                eprint()

                losses = self.validation_errors(predictors_test.cuda(), covariances_test.cuda())
                validation_errors.append(losses.tolist())
                validation_losses.append(validation_score)
                validation_stds.append(validation_std)
                optimization_losses.append(float(average_loss.data.numpy()[0]))


            else:
                keep_going = False
                eprint('Stopping because elements in the validation dataset have no neighbors.')


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


    def metadata(self):
        return {
            'algorithm': 'cello',
            'learning_rate': self.learning_rate,
            'alpha': self.alpha
        }


    def predict(self, queries):
        metric_matrix = self.theta_to_metric_matrix(self.theta)

        predictions = torch.zeros(len(queries),6,6, device='cuda')

        for i, x in enumerate(queries):
            distances = self.compute_distances(self.predictors, metric_matrix, x)

            predictions[i] = self.prediction_from_distances(self.covariances, distances)

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


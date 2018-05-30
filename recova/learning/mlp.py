
import sklearn
import torch
from torch.autograd import Variable
import torch.nn

from recova.learning.cello import kullback_leibler_pytorch
from recova.learning.model import CovarianceEstimationModel
from recova.util import eprint


class MlpModel(CovarianceEstimationModel):
    def __init__(self, device='cuda', learning_rate = 1e2, n_iterations=0, logging_rate=1000, alpha=0.0, convergence_window=50000, decay=1e-10):
        self.device = torch.device(device)
        self.hidden_sizes = [500, 250, 200, 400]
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.logging_rate = logging_rate
        self.alpha = alpha
        self.convergence_window = convergence_window
        self.weight_decay = decay


    def fit(self, predictors, covariances, train_set=None, test_set=None):

        if train_set and test_set:
            train_indices = train_set
            test_indices = test_set
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(list(range(len(predictors))), test_size=0.3)

        xs_train = Variable(torch.Tensor(predictors[train_indices])).to(self.device)
        ys_train = Variable(torch.Tensor(covariances[train_indices])).to(self.device)

        xs_test = Variable(torch.Tensor(predictors[test_indices])).to(self.device)
        ys_test = Variable(torch.Tensor(covariances[test_indices])).to(self.device)

        learning_run = self._fit(xs_train, ys_train, xs_test, ys_test)
        learning_run['train_set'] = train_indices
        learning_run['validation_set'] = test_indices

        return learning_run


    def _fit(self, xs_train, ys_train, xs_test, ys_test):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(xs_train[0]), self.hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.BatchNorm1d(self.hidden_sizes[0]),
            torch.nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.BatchNorm1d(self.hidden_sizes[1]),
            # torch.nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            # torch.nn.BatchNorm1d(self.hidden_sizes[2]),
            # torch.nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            # torch.nn.BatchNorm1d(self.hidden_sizes[3]),
            torch.nn.Linear(self.hidden_sizes[1], 6*6)
        ).to(self.device)

        self.best_loss = float('inf')
        n_iter_without_improvement = 0
        epoch = 0
        train_losses = []
        test_losses = []

        train_stds = []
        test_stds = []

        train_errors_log = []
        test_errors_log = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        while n_iter_without_improvement < self.convergence_window and (epoch < self.n_iterations or self.n_iterations == 0):

            loss = self._validate(xs_train, ys_train)

            self.model.zero_grad()
            loss.backward()

            optimizer.zero_grad()
            optimizer.step()

            # with torch.no_grad():
            #     for param in self.model.parameters():
            #         param.data -= self.learning_rate * param.grad

            test_loss = self._validate(xs_test, ys_test)

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                n_iter_without_improvement = 0
            else:
                n_iter_without_improvement += 1

            if epoch % self.logging_rate == 0:
                test_errors = self._validation_errors(xs_test, ys_test)

                eprint(test_errors)

                test_errors_log.append(test_errors.data.cpu().numpy().tolist())

                train_errors = self._validation_errors(xs_train, ys_train)

                eprint(train_errors)
                train_errors_log.append(train_errors.data.cpu().numpy().tolist())

                train_losses.append(loss.data.cpu().numpy().item())
                test_losses.append(test_loss.data.cpu().numpy().item())
                train_stds.append(train_errors.std().data.cpu().numpy().item())
                test_stds.append(test_errors.std().data.cpu().numpy().item())

                eprint('Train Loss: {:.8E}'.format(loss.data))
                eprint('Test loss:  {:.8E}'.format(test_loss.data))
                # self._dets(xs_test, ys_test)
                eprint('{} iterations without improvement (out of {})'.format(n_iter_without_improvement, self.convergence_window))
                eprint()

            epoch += 1

        return {
            'best_loss': self.best_loss.cpu().detach().numpy().item(),
            'metadata': {
                'algorithm': 'mlp',
                'learning_rate': self.learning_rate,
                'logging_rate': self.logging_rate,
                'n_iterations': self.n_iterations,
            },
            'optimization_errors': train_errors_log,
            'optimization_loss': train_losses,
            'validation_errors': test_errors_log,
            'validation_loss': test_losses,
            'validation_std': test_stds,
            'optimization_std': train_stds,
            'what': 'model learning',
        }


    def validate(self, predictors, covariances):
        xs = torch.Tensor(predictors).to(self.device)
        ys = torch.Tensor(covariance).to(self.device)

        return self._validate(xs, ys).mean().data[0]

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)

    def _validate(self, xs, ys):
        return self._validation_errors(xs,ys).mean()

    def _validation_errors(self, xs, ys):
        model_output = self.model(xs)
        reshaped_model_output = model_output.view(len(model_output), 6, 6)
        covariances_predicted = torch.bmm(reshaped_model_output, reshaped_model_output.transpose(1,2))

        errors = torch.sqrt((covariances_predicted - ys).pow(2.0).sum(dim=2).sum(dim=1))

        return errors

    def _dets(self, xs, ys):
        ys_predicted = self.model(xs)
        covariances_predicted = ys_predicted.view(len(ys_predicted), 6, 6)

        dets = torch.Tensor(len(covariances_predicted))
        for i, cov in enumerate(covariances_predicted):
            dets[i] = torch.det(cov)

        worst_cov = torch.argmin(torch.abs(dets))
        eprint(dets[worst_cov])
        eprint(covariances_predicted[worst_cov])


    def predict(self, predictors):
        predicted = self._predict(torch.Tensor(predictors).to(self.device))
        return predicted.detach().view(len(predicted),6,6).cpu().numpy()

    def _predict(self, xs):
        return self.model(xs)

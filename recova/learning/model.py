
import numpy as np
import torch

from recova.util import eprint, kullback_leibler


class CovarianceEstimationModel:
    def fit(self, xs, ys):
        raise NotImplementedError('CovarianceEstimationModels must implement fit method')

    def predict(self, xs):
        raise NotImplementedError('CovarianceEstimationModels must implement predict method')

    def validate(self, xs, ys):
        losses = self.validation_errors(xs, ys)
        return (losses.mean(), losses.std())

    def validation_errors(self, xs, ys):
        predictions = self.predict(xs)

        if torch.isnan(predictions).any():
            raise ValueError('NaNs found in provided predictions')

        losses = torch.zeros(len(predictions))

        total_loss = 0.
        for i in range(len(predictions)):
            losses[i] = torch.norm(ys[i] - predictions[i])

        return losses


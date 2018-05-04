
import numpy as np

from recova.util import eprint, kullback_leibler


class CovarianceEstimationModel:
    def fit(self, xs, ys):
        raise NotImplementedError('CovarianceEstimationModels must implement fit method')

    def predict(self, xs):
        raise NotImplementedError('CovarianceEstimationModels must implement predict method')

    def validate(self, xs, ys):
        """Given a validation set, outputs a loss."""
        predictions = self.predict(xs)

        if np.any(np.isnan(predictions)):
            raise ValueError('NaNs found in provided predictions')

        losses = []

        total_loss = 0.
        for i in range(len(predictions)):
            losses.append(kullback_leibler(ys[i], predictions[i]) + kullback_leibler(predictions[i], ys[i]))

        losses = np.array(losses)

        eprint('Validation score: {:.2E}'.format(np.mean(losses)))

        return (losses.mean(), losses.std())



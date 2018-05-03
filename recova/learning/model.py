
import numpy as np

from recova.util import eprint, kullback_leibler

import pdb


class CovarianceEstimationModel:
    def fit(self, xs, ys):
        raise NotImplementedError('CovarianceEstimationModels must implement fit method')

    def predict(self, xs):
        raise NotImplementedError('CovarianceEstimationModels must implement predict method')

    def validate(self, xs, ys):
        """Given a validation set, outputs a loss."""
        predictions = self.predict(xs)

        if np.any(np.isnan(predictions)):
            pdb.set_trace()
            raise ValueError('NaNs found in provided predictions')


        total_loss = 0.
        for i in range(len(predictions)):
            loss_of_i = kullback_leibler(ys[i], predictions[i])
            loss_of_i = kullback_leibler(predictions[i], ys[i])
            total_loss += loss_of_i

        eprint('Validation score: {:.2E}'.format(total_loss / len(xs)))
        eprint('Log Validation score: {:.2E}'.format(np.log(total_loss / len(xs))))

        return total_loss / len(xs)



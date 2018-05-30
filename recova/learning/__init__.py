
from recova.learning.cello import CelloCovarianceEstimationModel
from recova.learning.knn import KnnCovarianceEstimationModel
from recova.learning.mlp import MlpModel

def model_factory(model):
    if model == 'cello':
        return CelloCovarianceEstimationModel()
    elif model == 'knn':
        return KnnCovarianceEstimationModel()
    elif model == 'mlp':
        return MlpModel()
    else:
        raise ValueError('Invalid covariance estimation model {}'.format(model))


from recova.learning.cello import CelloCovarianceEstimationModel
from recova.learning.knn import KnnCovarianceEstimationModel

def model_factory(model):
    if model == 'cello':
        return CelloCovarianceEstimationModel()
    elif model == 'knn':
        return KnnCovarianceEstimationModel()
    else:
        raise ValueError('Invalid covariance estimation model {}'.format(model))

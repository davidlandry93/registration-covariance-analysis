import numpy as np

from recova.util import eprint, nearestPD, to_upper_triangular, upper_triangular_to_vector

class PreprocessingAlgorithm:
    def __init__(self):
        pass

    def process(self, m):
        raise RuntimeError('Preprocessing Algorithms must implement project')

    def unprocess(self, m):
        raise RuntimeError('Preprocessing algorithms must implement unprocess')

    def export(self):
        raise RuntimeError('Preprocessing algorithms must implement export_model')

    def import_model(self, model):
        raise RuntimeError('Preprocessing algorithms must implement import_model')

    def __repr__(self):
        raise RuntimeError('Preprocessing algorithms must implement __repr__')


class IdentityPreprocessing(PreprocessingAlgorithm):
    def __init__(self):
        pass

    def process(self, m):
        return m

    def unprocess(self, m):
        return m

    def export(self):
        return {
            'name': 'identity',
        }

    def import_model(self, model):
        pass

    def __repr__(self):
        return 'identity_preprocessing'


class TranslationOnlyPreprocessing(PreprocessingAlgorithm):
    def __init__(self):
        pass

    def process(self, m):
        return m[:, 0:3,
                 0:3]

    def unprocess(self, m):
        covariances = np.zeros((len(m), 6, 6))
        covariances[:,0:3,0:3] = m
        return covariances

    def export(self):
        return {
            'name': 'translation_only'
        }

    def import_model(self, model):
        pass

    def __repr__(self):
        return 'translation_only'


class CholeskyPreprocessing(PreprocessingAlgorithm):
    def __init__(self):
        pass

    def process(self, covariances):
        vectors = np.empty((len(covariances), 21))
        for i, cov in enumerate(covariances):
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                m = nearestPD(cov)
                L = np.linalg.cholesky(m)
            vectors[i] = upper_triangular_to_vector(L.T)
            eprint(vectors[i])

        return vectors

    def unprocess(self, m):
        covariances = np.empty((len(m)), 6, 6)

        for i, v in enumerate(m):
            up = to_upper_triangular(v)
            covariances[i] = np.dot(up, up.T)

            eprint(covariances[i])

        return covariances


    def export(self):
        return {
            'name': 'cholesky'
        }

    def import_model(self):
        pass

    def __repr__(self):
        'cholesky_preprocessing'






def preprocessing_factory(algo):
    if algo == 'identity':
        return IdentityPreprocessing()
    elif algo == 'translation_only':
        return TranslationOnlyPreprocessing()
    elif algo == 'cholesky':
        return CholeskyPreprocessing()
    else:
        raise RuntimeError('Unrecognized preprocessing algorithm {}'.format(algo))



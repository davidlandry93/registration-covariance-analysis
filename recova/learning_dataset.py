
import argparse
import json
import numpy as np
import pathlib


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def vectorize_covariance(cov_matrix):
    pd_matrix = nearestPD(cov_matrix)
    lower_triangular = np.linalg.cholesky(pd_matrix)

    vector_of_cov = []
    for i in range(6):
        for j in range(i + 1):
            vector_of_cov.append(lower_triangular[i, j])

    return vector_of_cov

def generate_examples_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='The name to give to the output files')
    parser.add_argument('n_examples', type=int, help='Number of examples to generate')
    parser.add_argument('--output', type=str, help='Where to store the examples', default='.')
    parser.add_argument('--input', type=str, help='Where the registration results are stored', default='.')
    args = parser.parse_args()

    np.set_printoptions(linewidth=120)

    output_path = pathlib.Path(args.output)

    for i in range(args.n_examples):
        with open('{}_{}/clusterings.json'.format(i+1, i)) as clusterings_file:
            clusterings = json.load(clusterings_file)
            cov_matrix = np.array(clusterings['data'][5]['covariance_of_central'])

            vector_of_cov = vectorize_covariance(cov_matrix)

            filename = 'example_{}_{}.json'.format(args.dataset_name, i)
            with (output_path / filename).open('w') as output_file:
                print(filename)
                json.dump(vector_of_cov, output_file)

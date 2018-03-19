
#include <iostream>

#include "greedy_seed_selection_algorithm.h"

namespace recova {

GreedySeedSelectionAlgorithm::GreedySeedSelectionAlgorithm(NaboAdapter& knn_algorithm, const int& k) : knn_algorithm(knn_algorithm), k(k) {}

int GreedySeedSelectionAlgorithm::select(std::shared_ptr<Eigen::MatrixXd>& dataset) {
    Eigen::MatrixXi indices(k, dataset->rows());
    Eigen::MatrixXd distances(k, dataset->rows());

    knn_algorithm.set_dataset(dataset);
    std::tie(indices, distances) = knn_algorithm.query(*dataset, k);
    std::vector<float> densities(dataset->cols());

    for(auto i = 0; i < dataset->cols(); i++) {
        densities[i] = distances(i, k-1) / (float) k;
    }

    int index_of_min = std::min_element(densities.begin(), densities.end()) - densities.begin();
    std::cerr << index_of_min << '\n';

    return index_of_min;
}

}

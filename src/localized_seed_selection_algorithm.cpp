
#include "localized_seed_selection_algorithm.h"

namespace recova {

LocalizedSeedSelectionAlgorithm::LocalizedSeedSelectionAlgorithm(NaboAdapter& knn_algorithm, const Eigen::VectorXd& location_of_search, const int& n_seeds_to_consider, const int& k) :
        knn_algorithm(knn_algorithm),
        location_of_search(location_of_search),
        n_seeds_to_consider(n_seeds_to_consider), k(k) {}

int LocalizedSeedSelectionAlgorithm::select(std::shared_ptr<Eigen::MatrixXd>& dataset) {
    Eigen::MatrixXi indices(n_seeds_to_consider, 1);
    Eigen::MatrixXd distances(n_seeds_to_consider, 1);

    knn_algorithm.set_dataset(dataset);

    std::tie(indices, distances) =
        knn_algorithm.query(location_of_search, n_seeds_to_consider);

    std::vector<int> vec_of_indices(indices.data(),
                                    indices.data() + indices.size());

    Eigen::MatrixXd search_result = knn_algorithm.get_ids_from_dataset(
        vec_of_indices.begin(), vec_of_indices.end());
    indices.resize(k, search_result.cols());
    distances.resize(k, search_result.cols());

    std::tie(indices, distances) = knn_algorithm.query(search_result, k);

    int min_index = 0;
    auto min_distance = std::numeric_limits<double>::infinity();
    for (auto i = 0; i < n_seeds_to_consider; i++) {
        double dist_of_current = distances(k - 1, i);
        if (dist_of_current < min_distance) {
            min_index = i;
            min_distance = distances(k - 1, i);
        }
    }

    return min_index;
}

}


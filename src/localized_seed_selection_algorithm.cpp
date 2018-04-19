
#include <iostream>

#include "localized_seed_selection_algorithm.h"
#include "pointcloud_logger_locator.h"

namespace recova {

LocalizedSeedSelectionAlgorithm::LocalizedSeedSelectionAlgorithm(
    NaboAdapter& knn_algorithm, const Eigen::VectorXd& location_of_search,
    const int& n_seeds_to_consider, const int& k)
    : knn_algorithm(knn_algorithm),
      location_of_search(location_of_search),
      n_seeds_to_consider(n_seeds_to_consider),
      k(k) {}

int LocalizedSeedSelectionAlgorithm::select(
    std::shared_ptr<Eigen::MatrixXd>& dataset) {
    PointcloudLoggerLocator logger_locator;
    PointcloudLogger& logger = logger_locator.get();


    Eigen::MatrixXi neighbors_of_seed_indices(n_seeds_to_consider, 1);
    Eigen::MatrixXd neighbors_of_seed_distances(n_seeds_to_consider, 1);

    knn_algorithm.set_dataset(dataset);

    logger.log_6d("location_of_search", location_of_search);

    std::tie(neighbors_of_seed_indices, neighbors_of_seed_distances) =
        knn_algorithm.query(location_of_search, n_seeds_to_consider);

    std::vector<int> vec_of_indices(neighbors_of_seed_indices.data(),
                                    neighbors_of_seed_indices.data() + neighbors_of_seed_indices.size());

    Eigen::MatrixXd search_result = knn_algorithm.get_ids_from_dataset(
        vec_of_indices.begin(), vec_of_indices.end());
    Eigen::MatrixXi neighbors_of_considered_indices(k, search_result.cols());
    Eigen::MatrixXd neighbors_of_considered_distances(k, search_result.cols());

    logger.log_6d("considered_points", search_result);

    std::tie(neighbors_of_considered_indices, neighbors_of_considered_distances) =
        knn_algorithm.query(search_result, k);

    int min_index = 0;
    auto min_distance = std::numeric_limits<double>::infinity();
    for (auto i = 0; i < n_seeds_to_consider; i++) {
        double dist_of_current = neighbors_of_considered_distances(k - 1, i);
        if (dist_of_current < min_distance) {
            min_index = i;
            min_distance = dist_of_current;
        }
    }

    return neighbors_of_seed_indices(min_index);
}
}

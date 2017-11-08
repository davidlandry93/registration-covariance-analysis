
#include <deque>
#include <memory>
#include <tuple>

#include <Eigen/Core>

#include "centered_clustering.h"

namespace recova {
int point_closest_to_center(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center) {
  Eigen::Matrix<double, 6, 1> center_query_matrix;
  center_query_matrix.col(0) = center;

  Eigen::MatrixXi indices;
  Eigen::MatrixXd distances;
  std::tie(indices, distances) = knn_algorithm.query(center, 1);

  return indices(0);
}


std::set<int> find_viable_cluster(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center) {
  return std::set<int>();
}


std::set<int> cluster_around(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center, const int& n, const double& radius) {
  std::deque<int> to_eval;
  std::set<int> cluster;

  to_eval.push_back(point_closest_to_center(knn_algorithm, center));

  while(to_eval.size() > 0) {
    Eigen::MatrixXd query_points = knn_algorithm.get_ids_from_dataset(to_eval);

    Eigen::MatrixXi indices(n, query_points.cols());
    Eigen::MatrixXd distances(n, query_points.cols());
    std::tie(indices, distances) = knn_algorithm.query(query_points, n);

    int size_of_to_eval = to_eval.size();

    // For every query point that was in the query we made to the Kd-Tree.
    for(auto i = 0; i < size_of_to_eval; i++) {
      if(distances(n - 1, i) < radius) {
        // Query point has at least N neighbours within radius.
        auto insert_result = cluster.insert(to_eval.front());
        bool was_inserted = insert_result.second;

        if(was_inserted) {
          // The point was not previously in the cluster.
          // Insert the neighbours of this query point to the to_eval.
          for(auto j = 0; j < n; j++) {
            to_eval.push_back(indices(j,i));
          }
        }
      } else {
        // Query point did not have enough neighbours to be a core point.
        // It's neighbours are not added.
        cluster.insert(to_eval.front());
      }

      to_eval.pop_front();
    }
  }

  return cluster;
}


std::vector<int> potential_seeds(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& start_from, const int& n_seeds=10) {
  Eigen::MatrixXi indices(n_seeds, 1);
  Eigen::MatrixXd distances(n_seeds, 1);

  std::tie(indices, distances) = knn_algorithm.query(start_from, n_seeds);

  std::vector<int> potential_seeds_vector(n_seeds);
  for(auto i = 0; i < n_seeds; i++) {
    potential_seeds_vector[i] = indices(i, 0);
  }

  return potential_seeds_vector;
}

}

#ifndef RECOVA_CENTERED_CLUSTERING_H
#define RECOVA_CENTERED_CLUSTERING_H

#include <set>
#include <vector>

#include <Eigen/Core>

#include "nabo_adapter.h"

namespace recova {

  std::set<int> cluster_around(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center, const int& n, const double& radius);

  std::set<int> find_viable_cluster(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center);
  int point_closest_to_center(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center);
  std::vector<int> potential_seeds(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& start_from, const int& n_seed);

}

#endif

#ifndef RECOVA_CENTERED_CLUSTERING_H
#define RECOVA_CENTERED_CLUSTERING_H

#include <memory>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "nabo_adapter.h"

namespace recova {

std::set<int> cluster_with_seed(const NaboAdapter &knn_algorithm,
                                const Eigen::VectorXd &seed, const int &n,
                                const double &radius);

int point_closest_to_center(const NaboAdapter &knn_algorithm,
                            const Eigen::VectorXd &center);

std::vector<int> potential_seeds(const NaboAdapter &knn_algorithm,
                                 const Eigen::VectorXd &start_from,
                                 const int &n_seed);

Eigen::VectorXd find_best_seed(const NaboAdapter &knn_algorithm,
                               const Eigen::VectorXd &location_of_search,
                               const int &n_seeds_to_consider, const int &n);

std::set<int> run_centered_clustering(std::unique_ptr<Eigen::MatrixXd>&& dataset,
                                      const Eigen::VectorXd& center,
                                      int k,
                                      double radius);
}

#endif

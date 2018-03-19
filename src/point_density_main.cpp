
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <gflags/gflags.h>
#include "json.hpp"

#include "nabo_adapter.h"
#include "util.h"

using json = nlohmann::json;
using namespace recova;

std::vector<double> size_of_neighborhood(std::shared_ptr<Eigen::MatrixXd>& dataset, const int& k) {
  NaboAdapter knn_algorithm;

  knn_algorithm.set_dataset(dataset);

  Eigen::MatrixXd distances(k, dataset->cols());
  Eigen::MatrixXi indices(k, dataset->cols());
  std::tie(indices, distances) = knn_algorithm.query(*dataset, k);

  return flatten(distances.row(k - 1));
}

DEFINE_int32(k, 12, "Number of points that comprises the neighbourhood of a given point.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  json json_dataset;
  std::cin >> json_dataset;

  auto eigen_dataset = std::make_shared<Eigen::MatrixXd>(json_array_to_matrix(json_dataset));
  std::vector<double> density = size_of_neighborhood(eigen_dataset, FLAGS_k);

  std::transform(density.begin(), density.end(), density.begin(), [](double x){return 1.0 / x;});

  std::cout << json(density) << std::endl;

  return 0;
}

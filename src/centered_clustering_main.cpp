
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include <Eigen/Core>
#include <gflags/gflags.h>
#include "json.hpp"

#include "nabo_adapter.h"

using namespace recova;
using json = nlohmann::json;

template <class OutIt> void explode(std::string &input, char sep, OutIt out) {
  std::istringstream buffer(input);
  std::string temp;

  while (std::getline(buffer, temp, sep))
    *(out++) = temp;
}

Eigen::MatrixXd json_array_to_matrix(const json& array) {
  int rows = array[0].size();
  int columns = array.size();
  Eigen::MatrixXd eigen_matrix(rows, columns);

  // We transpose the matrix during the copy because libnabo has the points on columns.
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < columns; j++) {
      eigen_matrix(i,j) = array[j][i];
    }
  }

  return eigen_matrix;
}


int point_closest_to_center(const NaboAdapter& knn_algorithm, const Eigen::VectorXd& center) {
  Eigen::Matrix<double, 6, 1> center_query_matrix;
  center_query_matrix.col(0) = center;

  Eigen::MatrixXi indices;
  Eigen::MatrixXd distances;
  std::tie(indices, distances) = knn_algorithm.query(center, 1);

  return indices(0);
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

DEFINE_int32(n, 12, "Number of elements within radius a point needs to have to be a core point.");
DEFINE_double(radius, 1.0, "Radius within which a point need to have n points to be a core point.");
DEFINE_string(seed, "", "The initial location where to start the cluster. Comma separated list of values representing the vector.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  json json_dataset;
  std::cin >> json_dataset;

  std::unique_ptr<Eigen::MatrixXd> eigen_dataset(new Eigen::MatrixXd);
  *eigen_dataset = json_array_to_matrix(json_dataset);

  NaboAdapter knn_algorithm;
  knn_algorithm.set_dataset(std::move(eigen_dataset));

  Eigen::VectorXd center = Eigen::VectorXd::Zero(6);
  if(!FLAGS_seed.empty()) {
    std::vector<std::string> elements(6);
    explode(FLAGS_seed, ',', elements.begin());

    for(auto i = 0; i < 6; ++i) {
      center(i) = stof(elements[i]);
    }
  }
  auto cluster = cluster_around(knn_algorithm, center, FLAGS_n, FLAGS_radius);

  json output_document;
  for(auto elt : cluster) {
    output_document.push_back(elt);
  }

  std::cout << output_document << std::endl;

  return 0;
}

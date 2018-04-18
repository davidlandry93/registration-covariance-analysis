
#include <memory>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Core>

#include "centered_clustering.h"
#include "greedy_seed_selection_algorithm.h"
#include "grid_pointcloud_separator.h"
#include "lieroy/algebra_se3.hpp"
#include "nabo_adapter.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace recova;

template<typename It>
p::list to_list(const It begin, const It end) {
  p::list output;
  for(auto it = begin; it != end; std::next(it)) {
    output.append(*it);
  }

  return output;
}

Eigen::MatrixXd ndarray_to_eigen_matrix(const np::ndarray& np_matrix) {
  auto eigen_matrix = Eigen::MatrixXd(np_matrix.shape(0), np_matrix.shape(1));

  for(auto i = 0; i < np_matrix.shape(0); ++i) {
    for(auto j = 0; j < np_matrix.shape(1); ++j) {
      eigen_matrix(i,j) = p::extract<double>(np_matrix[i][j]);
    }
  }

  return eigen_matrix;
}

p::list eigen_matrix_to_list(const Eigen::MatrixXd& eigen_m) {
  p::list matrix;

  for(auto i = 0; i < eigen_m.rows(); ++i) {
    p::list row;

    for(auto j = 0; j < eigen_m.cols(); ++j) {
      row.append(eigen_m(i,j));
    }

    matrix.append(row);
  }

  return matrix;
}

np::ndarray eigen_matrix_to_ndarray(const Eigen::MatrixXd& eigen_m) {
  return np::array(eigen_matrix_to_list(eigen_m));
}

p::list centered_clustering(const np::ndarray& m, const p::list& seed, int k, double radius) {
  auto eigen_matrix = std::shared_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd);
  *eigen_matrix = ndarray_to_eigen_matrix(m);

  Eigen::Matrix<double,6,1> eigen_seed;
  eigen_seed << ndarray_to_eigen_matrix(np::array(seed));
  lieroy::AlgebraSE3<double> lie_seed(eigen_seed);

  NaboAdapter knn_algorithm;
  auto seed_selector = std::unique_ptr<SeedSelectionAlgorithm>(new GreedySeedSelectionAlgorithm(knn_algorithm, lie_seed, 12));

  std::set<int> clustering = run_centered_clustering(eigen_matrix, std::move(seed_selector),  k, radius);

  return to_list(clustering.begin(), clustering.end());
}

p::list grid_pointcloud_separator(const np::ndarray& m,
                                  double spanx, double spany, double spanz,
                                  int nx, int ny, int nz) {
  GridPointcloudSeparator separator(spanx, spany, spanz, nx, ny, nz);
  auto pointcloud = std::unique_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd(ndarray_to_eigen_matrix(m)));
  pointcloud->transposeInPlace();

  separator.set_pointcloud(std::move(pointcloud));
  auto bins = separator.separate();

  p::list list_of_bins;
  for(auto i = 0; i < nx; i++) {
    for(auto j = 0; j < ny; j++) {
      for(auto k = 0; k < nz; k++) {
        auto bin = bins.get({i,j,k});

        p::list list_of_points;
        for(auto point : bin) {
          p::list python_point;

          python_point.append(point[0]);
          python_point.append(point[1]);
          python_point.append(point[2]);

          list_of_points.append(python_point);
        }

        list_of_bins.append(np::array(list_of_points));
      }
    }
  }

  return list_of_bins;
}


BOOST_PYTHON_MODULE(recova_core) {
  np::initialize();

  p:def("centered_clustering", centered_clustering, "Compute a clustering centered around zero of an ndarray.");
  p::def("grid_pointcloud_separator", grid_pointcloud_separator, "Separate a ndarray of points in a grid.");
}

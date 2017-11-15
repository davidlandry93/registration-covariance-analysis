
#include <memory>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Core>

#include "centered_clustering.h"
#include "nabo_adapter.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace recova;

Eigen::MatrixXd ndarray_to_eigen_matrix(const np::ndarray& np_matrix) {
  auto eigen_matrix = Eigen::MatrixXd(np_matrix.shape(0), np_matrix.shape(1));

  for(auto i = 0; i < np_matrix.shape(0); ++i) {
    for(auto j = 0; j < np_matrix.shape(1); ++j) {
      eigen_matrix(i,j) = p::extract<double>(np_matrix[i][j]);
    }
  }

  return eigen_matrix;
}

np::ndarray eigen_matrix_to_ndarray(const Eigen::MatrixXd& eigen_m) {
  p::list matrix;

  for(auto i = 0; i < eigen_m.rows(); ++i) {
    p::list row;

    for(auto j = 0; j < eigen_m.cols(); ++j) {
      row.append(eigen_m(i,j));
    }

    matrix.append(row);
  }

  return np::array(matrix);
}

np::ndarray centered_clustering(const np::ndarray& m) {
  auto eigen_matrix = std::unique_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd);
  *eigen_matrix = ndarray_to_eigen_matrix(m);

  std::set<int> clustering = run_centered_clustering(std::move(eigen_matrix), Eigen::VectorXd(3), 12, 1.0);

  return np::array(p::list());
}


BOOST_PYTHON_MODULE(recova_core) {
  np::initialize();

  p::def("centered_clustering", centered_clustering, "Compute a clustering centered around zero of an ndarray.");
}


#include <iostream>

#include "nabo_adapter.h"

using namespace Nabo;
using namespace Eigen;

namespace recova {

  void NaboAdapter::set_dataset(std::unique_ptr<Eigen::MatrixXd>&& p_dataset) {
    dataset = std::move(p_dataset);
    nns = std::unique_ptr<NNSearchD>(NNSearchD::createKDTreeLinearHeap(*dataset));
  }

  std::pair<MatrixXi, MatrixXd> NaboAdapter::query(const MatrixXd& query_points, const int& k) const {
    MatrixXi indices(k, query_points.cols());
    MatrixXd distances(k, query_points.cols());

    nns->knn(query_points, indices, distances, k, 0, NNSearchD::SORT_RESULTS);

    return std::make_pair(indices, distances);
  }

  Eigen::MatrixXd NaboAdapter::get_ids_from_dataset(const std::deque<int>& ids) const {
    int n_points = ids.size();
    Eigen::MatrixXd points(dataset->rows(), n_points);

    for(auto i = 0; i < n_points; i++) {
      points.col(i) = dataset->col(ids[i]);
    }

    return points;
  }
}

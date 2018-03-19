
#include <iostream>

#include "nabo_adapter.h"

using namespace Nabo;
using namespace Eigen;

namespace recova {

NaboAdapter::NaboAdapter() : dataset(nullptr) {}

NaboAdapter::NaboAdapter(NaboAdapter& lhs) : dataset(nullptr) {
    if (lhs.dataset != nullptr) {
        set_dataset(lhs.dataset);
    }
}

Eigen::VectorXd NaboAdapter::get_id_from_dataset(const int& index) const {
    return dataset->col(index);
}

void NaboAdapter::set_dataset(const Eigen::MatrixXd& p_dataset) {
    auto dataset = std::make_shared<Eigen::MatrixXd>(p_dataset);
    set_dataset(dataset);
}

void NaboAdapter::set_dataset(std::shared_ptr<Eigen::MatrixXd>& p_dataset) {
    dataset = p_dataset;
    nns = std::unique_ptr<NNSearchD>(NNSearchD::createKDTreeLinearHeap(*dataset));
}

std::pair<MatrixXi, MatrixXd> NaboAdapter::query(const MatrixXd& query_points,
                                                 const int& k) const {
    MatrixXi indices(k, query_points.cols());
    MatrixXd distances(k, query_points.cols());

    nns->knn(query_points, indices, distances, k, 0, NNSearchD::SORT_RESULTS);

    return std::make_pair(indices, distances);
}

Eigen::MatrixXd NaboAdapter::get_ids_from_dataset(
    const std::deque<int>& ids) const {
    int n_points = ids.size();
    Eigen::MatrixXd points(dataset->rows(), n_points);

    for (auto i = 0; i < n_points; i++) {
        points.col(i) = dataset->col(ids[i]);
    }

    return points;
}

}

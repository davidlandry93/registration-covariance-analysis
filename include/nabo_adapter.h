#ifndef RECOVA_NABO_ADAPTER_H
#define RECOVA_NABO_ADAPTER_H

#include <deque>
#include <memory>
#include <utility>

#include <nabo/nabo.h>
#include <Eigen/Core>

namespace recova {
class NaboAdapter {
  public:
    void set_dataset(std::unique_ptr<Eigen::MatrixXd>&& dataset);

    void set_dataset(const Eigen::MatrixXd& p_dataset);
    std::pair<Eigen::MatrixXi, Eigen::MatrixXd> query(
        const Eigen::MatrixXd& query_points, const int& n_neighbors) const;
    Eigen::VectorXd get_id_from_dataset(const int& index) const;
    Eigen::MatrixXd get_ids_from_dataset(const std::deque<int>& ids) const;

    template <typename Iterator>
    Eigen::MatrixXd get_ids_from_dataset(const Iterator& begin,
                                         const Iterator& end) const {
        Eigen::MatrixXd points(dataset->rows(), std::distance(begin, end));

        int n_points = std::distance(begin, end);
        for (auto it = begin; it != end; it = std::next(it)) {
            points.col(std::distance(begin, it)) = dataset->col(*it);
        }

        return points;
    }

  private:
    std::unique_ptr<Eigen::MatrixXd> dataset;
    std::unique_ptr<Nabo::NNSearchD> nns;
};
}

#endif

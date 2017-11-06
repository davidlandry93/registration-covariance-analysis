
#include <deque>
#include <memory>
#include <utility>

#include <Eigen/Core>
#include <nabo/nabo.h>


namespace recova {
  class NaboAdapter {
  public:
    void set_dataset(std::unique_ptr<Eigen::MatrixXd>&& dataset);
    std::pair<Eigen::MatrixXi, Eigen::MatrixXd> query(const Eigen::MatrixXd& query_points, const int& n_neighbors) const;
    Eigen::MatrixXd get_ids_from_dataset(const std::deque<int>& ids) const;
  private:
    std::unique_ptr<Eigen::MatrixXd> dataset;
    std::unique_ptr<Nabo::NNSearchD> nns;
  };
}

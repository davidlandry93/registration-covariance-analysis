#ifndef RECOVA_SEED_SELECTION_ALGORITHM_H
#define RECOVA_SEED_SELECTION_ALGORITHM_H

#include <Eigen/Core>
#include <memory>

namespace recova {

class SeedSelectionAlgorithm {
  public:
    virtual int select(std::shared_ptr<Eigen::MatrixXd>& dataset)=0;
    virtual ~SeedSelectionAlgorithm() = default;

  private:
};
}

#endif

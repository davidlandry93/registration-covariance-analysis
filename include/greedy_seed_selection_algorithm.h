#ifndef RECOVA_GREEDY_SEED_SELECTION_ALGORITHM_H
#define RECOVA_GREEDY_SEED_SELECTION_ALGORITHM_H


#include "nabo_adapter.h"
#include "seed_selection_algorithm.h"

#include "lieroy/algebra_se3.hpp"

namespace recova {

class GreedySeedSelectionAlgorithm : public SeedSelectionAlgorithm {
  public:
    GreedySeedSelectionAlgorithm(NaboAdapter& knn_algorithm, const lieroy::AlgebraSE3<double>& seed, const int& k);
    int select(std::shared_ptr<Eigen::MatrixXd>& dataset) override;

  private:
    NaboAdapter knn_algorithm;
    lieroy::AlgebraSE3<double> seed;
    int k;
};
}

#endif

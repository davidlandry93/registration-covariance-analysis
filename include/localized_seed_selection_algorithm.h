#ifndef RECOVA_LOCALIZED_SEED_SELECTION_ALGORITHM_H
#define RECOVA_LOCALIZED_SEED_SELECTION_ALGORITHM_H

#include <memory>

#include "nabo_adapter.h"
#include "seed_selection_algorithm.h"

namespace recova {

class LocalizedSeedSelectionAlgorithm : public SeedSelectionAlgorithm {
  public:
    LocalizedSeedSelectionAlgorithm(NaboAdapter& knn_algorithm, const Eigen::VectorXd& location_of_search, const int& n_seeds_to_consider, const int& k);
    int select(std::shared_ptr<Eigen::MatrixXd>& dataset) override;

  private:
    NaboAdapter knn_algorithm;
    Eigen::VectorXd location_of_search;
    int n_seeds_to_consider;
    int k;
};
}

#endif

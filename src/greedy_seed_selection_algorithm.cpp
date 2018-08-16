
#include <iostream>
#include <glog/logging.h>

#include "greedy_seed_selection_algorithm.h"
#include "lieroy/algebra_se3.hpp"
#include "lieroy/se3.hpp"

using namespace lieroy;

namespace recova {


GreedySeedSelectionAlgorithm::GreedySeedSelectionAlgorithm(NaboAdapter& knn_algorithm, const AlgebraSE3<double>& seed, const int& k) : knn_algorithm(knn_algorithm), seed(seed),  k(k) {}

int GreedySeedSelectionAlgorithm::select(std::shared_ptr<Eigen::MatrixXd>& dataset) {
    VLOG(100) << "Selecting seed with GreedySeedSelectionAlgorithm...";

    Eigen::MatrixXi indices(k, dataset->rows());
    Eigen::MatrixXd distances(k, dataset->rows());

    knn_algorithm.set_dataset(dataset);
    std::tie(indices, distances) = knn_algorithm.query(*dataset, k);
    std::vector<float> densities(dataset->cols());

    VLOG(90) << "KNN query distances has " << distances.rows() << " rows and " << distances.cols() << " columns." << '\n';

    for(auto i = 0; i < dataset->cols(); i++) {
        float density = distances(k-1, i);
        densities[i] = density;
    }

    // std::sort(densities.begin(), densities.end(),
    //           [](const std::tuple<int,float>& a, const std::tuple<int,float>& b) -> bool {
    //               return std::get<1>(a) > std::get<1>(b);
    //           });

    // int index_to_return = -1;
    // for(auto i = 0; i < densities.size(); i++) {
    //     // auto delta = seed.as_vector() - dataset->col(std::get<0>(densities[i]));

    //     if(i < 100) {
    //         auto GT = seed.exp();
    //         Eigen::Matrix<double,6,1> m_of_t;
    //         m_of_t << dataset->col(std::get<0>(densities[0]));
    //         AlgebraSE3<double> algebra_T(m_of_t);

    //         auto se3_delta = GT.inv() * algebra_T.exp();

    //         std::cerr << "Delta transofmration" << std::endl;
    //         std::cerr << se3_delta << std::endl;

    //     }

    // }

    int to_return = std::min_element(densities.begin(), densities.end()) - densities.begin();
    VLOG(90) << "Returning index: " << to_return << " with value " << densities[to_return];
    VLOG(90) << "Value of returned index: " << densities[to_return];


    VLOG(100) << "Done selecting seed with GreedySeedSelectionAlgorithm...";
    return to_return;
}

}

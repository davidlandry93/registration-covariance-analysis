
#include <iostream>
#include <string>

#include <gflags/gflags.h>
#include "json.hpp"

#include "centered_clustering.h"
#include "greedy_seed_selection_algorithm.h"
#include "localized_seed_selection_algorithm.h"
#include "nabo_adapter.h"
#include "seed_selection_algorithm.h"
#include "util.h"


using namespace recova;
using json = nlohmann::json;

template <class OutIt> void explode(std::string &input, char sep, OutIt out) {
  std::istringstream buffer(input);
  std::string temp;

  while (std::getline(buffer, temp, sep))
    *(out++) = temp;
}

Eigen::VectorXd parse_seed(std::string& seed_str) {
    Eigen::VectorXd center = Eigen::VectorXd::Zero(6);
    if(!seed_str.empty()) {
        std::vector<std::string> elements(6);
        explode(seed_str, ',', elements.begin());

        for(auto i = 0; i < 6; ++i) {
            center(i) = stof(elements[i]);
        }
    }

    return center;
}

DEFINE_int32(k, 12, "Number of elements within radius a point needs to have to be a core point.");
DEFINE_double(radius, 1.0, "Radius within which a point need to have n points to be a core point.");
DEFINE_string(seed, "", "The initial location where to start the cluster. Comma separated list of values representing the vector.");
DEFINE_int32(n_seed_init, 100, "The number of seeds close the the ground truth to consider during initialization.");
DEFINE_string(seed_selector, "localized", "The seed selection strategy.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  json json_dataset;
  std::cin >> json_dataset;

  std::shared_ptr<Eigen::MatrixXd> eigen_dataset(new Eigen::MatrixXd);
  *eigen_dataset = json_array_to_matrix(json_dataset);

  auto center = parse_seed(FLAGS_seed);


  std::unique_ptr<SeedSelectionAlgorithm> seed_selector;
  NaboAdapter knn_algorithm;
  if(FLAGS_seed_selector == "localized") {
      seed_selector.reset(new LocalizedSeedSelectionAlgorithm(knn_algorithm, center, FLAGS_n_seed_init, FLAGS_k));
  } else if (FLAGS_seed_selector == "greedy") {
      seed_selector.reset(new GreedySeedSelectionAlgorithm(knn_algorithm, FLAGS_k));
  }

  auto cluster = run_centered_clustering(eigen_dataset, std::move(seed_selector), FLAGS_k, FLAGS_radius);

  json output_document;
  for(auto elt : cluster) {
    output_document.push_back(elt);
  }

  std::cout << output_document << std::endl;

  return 0;
}

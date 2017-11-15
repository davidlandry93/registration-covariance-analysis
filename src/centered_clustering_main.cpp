
#include <iostream>
#include <string>

#include <gflags/gflags.h>
#include "json.hpp"

#include "centered_clustering.h"
#include "nabo_adapter.h"
#include "util.h"


using namespace recova;
using json = nlohmann::json;

template <class OutIt> void explode(std::string &input, char sep, OutIt out) {
  std::istringstream buffer(input);
  std::string temp;

  while (std::getline(buffer, temp, sep))
    *(out++) = temp;
}

DEFINE_int32(k, 12, "Number of elements within radius a point needs to have to be a core point.");
DEFINE_double(radius, 1.0, "Radius within which a point need to have n points to be a core point.");
DEFINE_string(seed, "", "The initial location where to start the cluster. Comma separated list of values representing the vector.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  json json_dataset;
  std::cin >> json_dataset;

  std::unique_ptr<Eigen::MatrixXd> eigen_dataset(new Eigen::MatrixXd);
  *eigen_dataset = json_array_to_matrix(json_dataset);


  Eigen::VectorXd center = Eigen::VectorXd::Zero(6);
  if(!FLAGS_seed.empty()) {
    std::vector<std::string> elements(6);
    explode(FLAGS_seed, ',', elements.begin());

    for(auto i = 0; i < 6; ++i) {
      center(i) = stof(elements[i]);
    }
  }

  auto cluster = run_centered_clustering(std::move(eigen_dataset), center, FLAGS_k, FLAGS_radius);

  json output_document;
  for(auto elt : cluster) {
    output_document.push_back(elt);
  }

  std::cout << output_document << std::endl;

  return 0;
}


#include <fstream>
#include <iostream>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "json.hpp"

#include "centered_clustering.h"
#include "greedy_seed_selection_algorithm.h"
#include "lieroy/algebra_se3.hpp"
#include "localized_seed_selection_algorithm.h"
#include "nabo_adapter.h"
#include "null_pointcloud_logger.h"
#include "pointcloud_io.hpp"
#include "pointcloud_logger_locator.h"
#include "seed_selection_algorithm.h"
#include "util.h"
#include "xyz_pointcloud_logger.h"


using namespace lieroy;
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
DEFINE_int32(n_seed_init, 20, "The number of seeds close the the ground truth to consider during initialization.");
DEFINE_string(seed_selector, "localized", "The seed selection strategy.");
DEFINE_bool(pointcloud_log, false, "Export pointcloud logging");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  json json_dataset;
  std::cin >> json_dataset;

  std::shared_ptr<Eigen::MatrixXd> eigen_dataset(new Eigen::MatrixXd);
  *eigen_dataset = json_array_to_matrix(json_dataset).transpose();

  VLOG(10) << "Matrix has " << eigen_dataset->rows() << " rows and " << eigen_dataset->cols() << " columns." << '\n';

  PointcloudLoggerLocator locator;
  std::unique_ptr<PointcloudLogger> logger;
  if(FLAGS_pointcloud_log) {
      logger.reset(new XyzPointcloudLogger("."));
  } else {
      logger.reset(new NullPointcloudLogger);
  }
  locator.set(std::move(logger));


  if(FLAGS_pointcloud_log) {
      std::ofstream output_stream;

      output_stream.open("input_cloud_translation.xyz");
      eigen_to_xyz<double>(eigen_dataset->topRows(3), output_stream);
      output_stream.close();

      output_stream.open("input_cloud_rotation.xyz");
      eigen_to_xyz<double>(eigen_dataset->bottomRows(3), output_stream);
      output_stream.close();
  }

  auto center = parse_seed(FLAGS_seed);



  SeedSelectionAlgorithm* seed_selector;
  NaboAdapter knn_algorithm;
  if(FLAGS_seed_selector == "localized") {
      seed_selector = new LocalizedSeedSelectionAlgorithm(knn_algorithm, center, FLAGS_n_seed_init, FLAGS_k);
  } else if (FLAGS_seed_selector == "greedy") {
      Eigen::Matrix<double,6,1> eigen_center;
      eigen_center << center;
      seed_selector = new GreedySeedSelectionAlgorithm(knn_algorithm, AlgebraSE3<double>(eigen_center), FLAGS_k);
  } else {
      throw std::runtime_error("Bad choice of seed selector not found");
  }

  auto cluster = run_centered_clustering(eigen_dataset, *seed_selector, FLAGS_k, FLAGS_radius, FLAGS_pointcloud_log);

  delete seed_selector;

  json output_document;
  for(auto elt : cluster) {
    output_document.push_back(elt);
  }

  std::cout << output_document << std::endl;

  return 0;
}

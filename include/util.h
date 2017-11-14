
#include <vector>

#include <Eigen/Core>
#include "json.hpp"

namespace recova {
  Eigen::MatrixXd json_array_to_matrix(const nlohmann::json& array);
  std::vector<double> flatten(const Eigen::MatrixXd& m);
}

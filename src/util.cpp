
#include "json.hpp"
#include "util.h"

using json = nlohmann::json;

namespace recova {
  Eigen::MatrixXd json_array_to_matrix(const json& array) {
    int rows = array.size();
    int columns = array[0].size();
    Eigen::MatrixXd eigen_matrix(rows, columns);

    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < columns; j++) {
        eigen_matrix(i,j) = array[i][j];
      }
    }

    return eigen_matrix;
  }

  std::vector<double> flatten(const Eigen::MatrixXd& m) {
    return std::vector<double>(m.data(), m.data() + m.size());
  }
}

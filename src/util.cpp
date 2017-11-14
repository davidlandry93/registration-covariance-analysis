
#include "json.hpp"
#include "util.h"

using json = nlohmann::json;

namespace recova {
  Eigen::MatrixXd json_array_to_matrix(const json& array) {
    int rows = array[0].size();
    int columns = array.size();
    Eigen::MatrixXd eigen_matrix(rows, columns);

    // We transpose the matrix during the copy because libnabo has the points on columns.
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < columns; j++) {
        eigen_matrix(i,j) = array[j][i];
      }
    }

    return eigen_matrix;
  }

  std::vector<double> flatten(const Eigen::MatrixXd& m) {
    return std::vector<double>(m.data(), m.data() + m.size());
  }
}

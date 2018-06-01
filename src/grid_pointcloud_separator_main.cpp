
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <gflags/gflags.h>

#include "grid_pointcloud_separator.h"
#include "json.hpp"
#include "util.h"

using namespace recova;
using namespace std;

using json = nlohmann::json;

DEFINE_double(spanx, 1.0, "The size of the voxellized area in x");
DEFINE_double(spany, 1.0, "The size of the voxellized area in y");
DEFINE_double(spanz, 1.0, "The size of the voxellized area in z");
DEFINE_int32(nx, 10, "Number of cells in the x axis");
DEFINE_int32(ny, 10, "Number of cells in the y axis");
DEFINE_int32(nz, 10, "Number of cells in the z axis");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  json input;
  cin >> input;


  unique_ptr<Eigen::MatrixXd> m(new Eigen::MatrixXd());
  *m = json_array_to_matrix(input).transpose();

  GridPointcloudSeparator separator(FLAGS_spanx, FLAGS_spany, FLAGS_spanz, FLAGS_nx, FLAGS_ny, FLAGS_nz);
  separator.set_pointcloud(move(m));
  auto bins = separator.separate();

  json output;
  for(auto i = 0; i < FLAGS_nx; i++) {
    for(auto j = 0; j < FLAGS_ny; j++) {
      for(auto k = 0; k < FLAGS_nz; k++) {
        auto bin = bins.get({i,j,k});

        json json_bin = json::array();
        for(auto point : bin) {
          json row;
          row.push_back(point(0));
          row.push_back(point(1));
          row.push_back(point(2));
          json_bin.push_back(row);
        }
        output.push_back(json_bin);
      }
    }
  }

  cout << output;
  return 0;
}

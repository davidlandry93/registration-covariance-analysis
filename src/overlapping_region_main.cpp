#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

#include <gflags/gflags.h>
#include <Eigen/Core>

#include "json.hpp"
#include "nabo_adapter.h"
#include "util.h"

DEFINE_string(reading, "", "Json file containing the reading datapoints");
DEFINE_string(reference, "", "Json file containing the reference datapoints");
DEFINE_string(
    t,
    "[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]",
    "A 4x4 transformation to apply to the reading");
DEFINE_double(radius, 0.1, "Maximum distance between overlapping points.");

using json = nlohmann::json;

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string usage =
        "overlapping_region -reading <reading_file.json> -reference "
        "<reference_file.json>";

    if (FLAGS_reading.empty() || FLAGS_reference.empty()) {
        std::cout << usage << '\n';
        return 0;
    }

    json t_json = json::parse(FLAGS_t);
    auto t = recova::json_array_to_matrix(t_json);

    json reading_json, reference_json;

    std::ifstream ifs(FLAGS_reading);
    ifs >> reading_json;
    ifs.close();
    ifs.open(FLAGS_reference);
    ifs >> reference_json;

    Eigen::MatrixXd reference = recova::json_array_to_matrix(reference_json);

    Eigen::MatrixXd reading = recova::json_array_to_matrix(reading_json);

    recova::NaboAdapter nabo;
    nabo.set_dataset(reference);

    Eigen::MatrixXi indices;
    Eigen::MatrixXd distances;

    std::tie(indices, distances) = nabo.query(reading, 1);

    Eigen::MatrixXd overlapping_points(3, reference.cols() + reading.cols());
    int n_overlapping_points = 0;


    for (auto i = 0; i < indices.cols(); ++i) {
        if (distances(i) < FLAGS_radius) {
            overlapping_points.col(n_overlapping_points++) = reading.col(i);
            overlapping_points.col(n_overlapping_points++) =
                reference.col(indices(i));
        }
    }

    overlapping_points.conservativeResize(3, n_overlapping_points);

    json output = json::array();

    for(auto i = 0; i < n_overlapping_points; ++i) {
        json data_row = json::array();

        data_row.push_back(overlapping_points(0,i));
        data_row.push_back(overlapping_points(1,i));
        data_row.push_back(overlapping_points(2,i));

        output.push_back(data_row);
    }

    std::cout << output;

    return 0;
}

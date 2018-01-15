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

DEFINE_double(radius, 0.1, "Maximum distance between overlapping points.");

using json = nlohmann::json;


Eigen::MatrixXd overlapping_points(const Eigen::MatrixXd& query, const Eigen::MatrixXd& reference) {
    recova::NaboAdapter nabo;
    nabo.set_dataset(reference);

    Eigen::MatrixXi indices;
    Eigen::MatrixXd distances;

    std::tie(indices, distances) = nabo.query(query, 1);

    Eigen::MatrixXd overlapping_points(3, query.cols());
    int n_overlapping_points = 0;

    for (auto i = 0; i < indices.cols(); ++i) {
        if (distances(i) < FLAGS_radius) {
            overlapping_points.col(n_overlapping_points++) = query.col(i);
        }
    }

    overlapping_points.conservativeResize(Eigen::NoChange, n_overlapping_points);
    return overlapping_points;
}

void add_matrix_to_json(const Eigen::MatrixXd& m, json& output) {
    for (auto i = 0; i < m.cols(); ++i) {
        json data_row = json::array();

        data_row.push_back(m(0, i));
        data_row.push_back(m(1, i));
        data_row.push_back(m(2, i));

        output.push_back(data_row);
    }
}


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string usage = "overlapping_region -radius <radius> ";

    json input;
    std::cin >> input;

    auto t = recova::json_array_to_matrix(input["t"]);
    Eigen::MatrixXd reference =
        recova::json_array_to_matrix(input["reference"]);
    Eigen::MatrixXd reading = recova::json_array_to_matrix(input["reading"]);

    Eigen::MatrixXd homogeneous_reading = Eigen::MatrixXd::Constant(4, reading.cols(), 1.0);
    homogeneous_reading.block(0,0, 3, reading.cols()) = reading;

    homogeneous_reading = t * homogeneous_reading;
    homogeneous_reading.conservativeResize(3, Eigen::NoChange);

    auto overlapping_from_reading = overlapping_points(homogeneous_reading, reference);
    auto overlapping_from_reference = overlapping_points(reference, homogeneous_reading);

    json output = json::array();
    add_matrix_to_json(overlapping_from_reference, output);
    add_matrix_to_json(overlapping_from_reading, output);


    std::cout << output << '\n';
    std::cerr << "Overlap: " << (double) output.size() / (double) (reference.cols() + reading.cols()) << '\n';

    return 0;
}

#include <iostream>

#include "json.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
    json input;
    std::cin >> input;

    for(auto point : input) {
        std::cout << point[0] << " " << point[1] << " " << point[2] << '\n';
    }

    return 0;
}

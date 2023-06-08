#include <filter/filter.hpp>
#include <cmath>
#include<iostream>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Error: Not enough arguments. Usage: " << argv[0] << " <input_file_path> <algorithm_type(gaussian/deriche/both)> <sigma> <visualization(yes/no)> <output_file_path1> [output_file_path2]" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string algorithm_type = argv[2];
    double sigma = std::stod(argv[3]);
    std::string visualization = argv[4];
    std::string output_file_path1 = argv[5];
    std::string output_file_path2 = "";

    if (argc == 7) {
        output_file_path2 = argv[6];
    }
    GaussianFilter filter;
    filter.compare_algorithms(input_filename, sigma, algorithm_type, visualization, output_file_path1, output_file_path2);
    return 0;
}

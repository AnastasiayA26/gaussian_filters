#ifndef GAUSSIANFILTER_FILTERG_HPP
#define GAUSSIANFILTER_FILTERG_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class GaussianFilter : public cv::Algorithm {
public:
    void recursiveGaussianDeriche(const cv::Mat& src, cv::Mat& dst, double sigma);

    cv::Mat read_input_data(const std::string& filename);

    cv::Mat gaussian_filter(const cv::Mat& input, double sigma);

    double calculateGaussianError(const cv::Mat& input, const cv::Mat& output);

    double calculateRecursiveGaussianDericheError(const cv::Mat& input, const cv::Mat& output);

    bool file_exists(const std::string& filename);

    void visualizeData(const std::vector<std::pair<double, double>>& data);

    std::vector<std::pair<double, double>> calculateDeviations(const cv::Mat& image, const std::string& mode, double sigma);

    void compare_algorithms(const std::string& input_filename, double sigma, const std::string& algorithm_type, const std::string& visualization, const std::string& output_file_path1, const std::string& output_file_path2);
};
#endif // GAUSSIANFILTER_FILTERG_HPP


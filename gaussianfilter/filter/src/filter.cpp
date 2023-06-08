#include <filter/filter.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <ctikz/ctikz.hpp>
#include <string>
#include <string.h>

#define mp make_pair


void GaussianFilter::recursiveGaussianDeriche(const cv::Mat& src, cv::Mat& dst, double sigma) {
    double alpha = 1.695 * sigma;
    double ema = std::exp(-alpha);
    double ema2 = std::exp(-2 * alpha);
    double k = (1 - ema) * (1 - ema) / (1 + (2 * alpha * ema) - ema2);
    double a1 = k * ema;
    double a2 = k * ema * ema;
    double a3 = k * ema2;
    double b1 = 2 * ema;
    double b2 = -ema2;

    dst.create(src.size(), src.type());
    cv::Mat temp(src.size(), src.type());

    // Apply the filter row-wise
    for (int y = 0; y < src.rows; y++) {
        double xm1 = src.at<double>(y, 0);
        double xm2 = src.at<double>(y, 1);
        double ym1 = a1 * xm1;
        double ym2 = a1 * xm2 + a2 * xm1;

        for (int x = 0; x < src.cols; x++) {
            double xn = src.at<double>(y, x);
            double yn = a1 * xn + b1 * ym1 + b2 * ym2;
            temp.at<double>(y, x) = yn;
            xm2 = xm1;
            xm1 = xn;
            ym2 = ym1;
            ym1 = yn;
        }

        double xp1 = src.at<double>(y, src.cols - 2);
        double xp2 = src.at<double>(y, src.cols - 1);
        double yp1 = 0.0;
        double yp2 = 0.0;

        for (int x = src.cols - 1; x >= 0; x--) {
            double xn = src.at<double>(y, x);
            double yn = a3 * xp1 + b1 * yp1 + b2 * yp2;
            dst.at<double>(y, x) = (temp.at<double>(y, x) + yn) / 2.0;
            xp2 = xp1;
            xp1 = xn;
            yp2 = yp1;
            yp1 = yn;
        }
    }

    // Apply the filter column-wise
    for (int x = 0; x < dst.cols; x++) {
        double xm1 = dst.at<double>(0, x);
        double xm2 = dst.at<double>(1, x);
        double ym1 = a1 * xm1;
        double ym2 = a1 * xm2 + a2 * xm1;

        for (int y = 0; y < dst.rows; y++) {
            double xn = dst.at<double>(y, x);
            double yn = a1 * xn + b1 * ym1 + b2 * ym2;
            temp.at<double>(y, x) = yn;
            xm2 = xm1;
            xm1 = xn;
            ym2 = ym1;
            ym1 = yn;
        }

        double yp1 = dst.at<double>(dst.rows - 2, x);
        double yp2 = dst.at<double>(dst.rows - 1, x);
        double xp1 = 0.0;
        double xp2 = 0.0;

        for (int y = dst.rows - 1; y >= 0; y--) {
            double xn = dst.at<double>(y, x);
            double yn = a3 * yp1 + b1 * xp1 + b2 * xp2;
            dst.at<double>(y, x) = (temp.at<double>(y, x) + yn) / 2.0;
            yp2 = yp1;
            yp1 = xn;
            xp2 = xp1;
            xp1 = yn;
        }
    }
}

cv::Mat GaussianFilter::read_input_data(const std::string& filename) {
    cv::Mat input_data = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (input_data.empty()) {
        std::cout << "Error opening input file!" << std::endl;
        exit(1);
    }
    cv::Mat normalized_input;
    input_data.convertTo(normalized_input, CV_64F, 1.0 / 255.0); // Convert to double precision and normalize
    return normalized_input;
}

cv::Mat GaussianFilter::gaussian_filter(const cv::Mat& input, double sigma) {
    int ksize = std::round(6 * sigma) + 1;
    if (ksize % 2 == 0) {
        ksize += 1;
    }
    cv::Mat kernel = cv::getGaussianKernel(ksize, sigma, CV_64F);
    double sum = cv::sum(kernel)[0];
    kernel /= sum;
    cv::Mat output;
    cv::filter2D(input, output, -1, kernel);

    return output;
}

double GaussianFilter::calculateGaussianError(const cv::Mat& input, const cv::Mat& output) {
    // Calculate the Mean Squared Error (MSE) between input and output
    cv::Mat error;
    cv::absdiff(input, output, error);
    error = error.mul(error);  // Element-wise square
    double mse = cv::sum(error)[0] / (input.rows * input.cols);
    return mse;
}

double GaussianFilter::calculateRecursiveGaussianDericheError(const cv::Mat& input, const cv::Mat& output) {
    // Calculate the Mean Squared Error (MSE) between input and output
    cv::Mat error;
    cv::absdiff(input, output, error);
    error = error.mul(error);  // Element-wise square
    double mse = cv::sum(error)[0] / (input.rows * input.cols);
    return mse;
}

bool GaussianFilter::file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}
void GaussianFilter::visualizeData(const std::vector<std::pair<double, double>>& data) {
    CTikz test;
    test.start_axis();
    std::ifstream file("data.txt");
    if (file.is_open()) {
        double x, y;
        std::vector<std::pair<double, double>> asd;
        while (file >> x >> y) {
            asd.push_back(std::make_pair(x, y));
        }
        file.close();

        FunctionStyle style;
        test.drawFunc(asd);
        test.end_axis();
        test.create_tikz_file("result");
    } else {
        std::cerr << "Failed to open the file." << std::endl;
    }

    std::cout << 1;
}

std::vector<std::pair<double, double>> GaussianFilter::calculateDeviations(const cv::Mat& image, const std::string& mode, double sigma) {
    cv::Mat deviationImage;
    if (mode == "both") {
        cv::Mat recursiveFiltered;
        recursiveGaussianDeriche(image, recursiveFiltered, sigma);
        cv::Mat gaussianFiltered = gaussian_filter(image, sigma);
        cv::absdiff(recursiveFiltered, gaussianFiltered, deviationImage);
    }
    else if (mode == "gaussian") {
        cv::Mat gaussianFiltered = gaussian_filter(image, sigma);
        cv::absdiff(image, gaussianFiltered, deviationImage);
    }
    else if (mode == "deriche") {
        cv::Mat dericheFiltered;
        recursiveGaussianDeriche(image, dericheFiltered, sigma);
        cv::absdiff(image, dericheFiltered, deviationImage);
    }
    std::vector<std::pair<double, double>>  deviationCoordinates;
    int centerRow = deviationImage.rows / 2;
    const double* deviationRow = deviationImage.ptr<double>(centerRow);
    for (int x = 1; x < deviationImage.cols; ++x) {
        double deviationX = (x - 1) * sigma;
        double deviationY = deviationRow[x] * sigma;
        deviationCoordinates.emplace_back(deviationX, deviationY);
    }
    return deviationCoordinates;
}

void GaussianFilter::compare_algorithms(const std::string & input_filename, double sigma, const std::string & algorithm_type, const std::string & visualization, const std::string & output_file_path1, const std::string & output_file_path2) {
    if (!file_exists(input_filename)) {
        std::cerr << "Error: Input file " << input_filename << " does not exist." << std::endl;
        return;
    }
    if (algorithm_type != "gaussian" && algorithm_type != "deriche" && algorithm_type != "both") {
        std::cerr << "Error: Invalid algorithm type. Must be 'gaussian', 'deriche', or 'both'." << std::endl;
        return;
    }
    if (visualization != "yes" && visualization != "no") {
        std::cerr << "Error: Invalid visualization option. Must be 'yes' or 'no'." << std::endl;
        return;
    }
    cv::Mat input = read_input_data(input_filename);
    cv::Mat cv_gaussian_output(input.size(), CV_64F);
    cv::Mat custom_gaussian_output(input.size(), CV_64F);
    std::vector<cv::Point2d> recursiveCoordinates;
    std::vector<cv::Point2d> gaussianCoordinates;

    if (algorithm_type == "gaussian") {
        custom_gaussian_output = gaussian_filter(input, sigma);

        if (!output_file_path1.empty()) {
            if (!file_exists(output_file_path1)) {
                cv::Mat output_image;
                custom_gaussian_output.convertTo(output_image, CV_64F);
                cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
                cv::imwrite(output_file_path1, output_image);
            }
            else {
                std::cerr << "Error: Output file path " << output_file_path1 << " already exists." << std::endl;
                return;
            }
        }

        double gaussian_error = calculateGaussianError(input, custom_gaussian_output);
        if (visualization == "yes") {
            std::vector<std::pair<double, double>>  deviations = calculateDeviations(input, algorithm_type, sigma);
            std::ofstream out_file("data.txt");
            for (const auto& deviation : deviations) {
                out_file << deviation.first << " " << deviation.second << std::endl;
            }
            out_file.close();
            visualizeData(deviations);
        }

        if (visualization == "no") {
            std::cout << "MSE Error for custom Gaussian filter: " << gaussian_error << std::endl;
        }
    }

    if (algorithm_type == "deriche") {
        cv::Mat recursive_gaussian_output(input.size(), CV_64F);
        recursiveGaussianDeriche(input, recursive_gaussian_output, sigma);

        if (!output_file_path1.empty()) {
            if (!file_exists(output_file_path1)) {
                cv::Mat output_image;
                recursive_gaussian_output.convertTo(output_image, CV_64F);
                cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
                cv::imwrite(output_file_path1, output_image);
            }
            else {
                std::cerr << "Error: Output file path " << output_file_path1 << " already exists." << std::endl;
                return;
            }
        }

        double recursive_gaussian_error = calculateRecursiveGaussianDericheError(input, recursive_gaussian_output);

        if (visualization == "yes") {
            std::vector<std::pair<double, double>> deviations = calculateDeviations(input, algorithm_type, sigma);
            std::ofstream out_file("data.txt");
            for (const auto& deviation : deviations) {
                out_file << deviation.first << " " << deviation.second << std::endl;
            }
            out_file.close();
            visualizeData(deviations);
        }

        if (visualization == "no") {
            std::cout << "MSE for Recursive Gaussian Deriche filter: " << recursive_gaussian_error << std::endl;
        }
    }

    if (algorithm_type == "both") {
        custom_gaussian_output = gaussian_filter(input, sigma);

        if (!output_file_path1.empty()) {
            if (!file_exists(output_file_path1)) {
                cv::Mat output_image;
                custom_gaussian_output.convertTo(output_image, CV_64F);
                cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
                cv::imwrite(output_file_path1, output_image);
            }
            else {
                std::cerr << "Error: Output file path " << output_file_path1 << " already exists." << std::endl;
                return;
            }
        }

        cv::Mat recursive_gaussian_output(input.size(), CV_64F);
        recursiveGaussianDeriche(input, recursive_gaussian_output, sigma);

        if (!output_file_path2.empty()) {
            if (!file_exists(output_file_path2)) {
                cv::Mat output_image;
                recursive_gaussian_output.convertTo(output_image, CV_64F);
                cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
                cv::imwrite(output_file_path2, output_image);
            }
            else {
                std::cerr << "Error: Output file path " << output_file_path2 << " already exists." << std::endl;
                return;
            }
        }

        double gaussian_error = calculateGaussianError(input, custom_gaussian_output);
        double recursive_gaussian_error = calculateRecursiveGaussianDericheError(input, recursive_gaussian_output);

        if (visualization == "yes") {
            std::vector<std::pair<double, double>> deviations = calculateDeviations(input, algorithm_type, sigma);
            std::ofstream out_file("data.txt");
            for (const auto& deviation : deviations) {
                out_file << deviation.first << " " << deviation.second << std::endl;
            }
            out_file.close();
            visualizeData(deviations);
        }
        if (visualization == "no") {
            std::cout << "MSE Error for custom Gaussian filter: " << gaussian_error << std::endl;
            std::cout << "MSE for Recursive Gaussian Deriche filter: " << recursive_gaussian_error << std::endl;
            std::cout << "Difference between Gaussian filter and Gaussian Deriche filter: " << std::abs(gaussian_error - recursive_gaussian_error) << std::endl;
        }
    }
}
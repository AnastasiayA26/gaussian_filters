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
    input_data.convertTo(normalized_input, CV_64F); // Convert to double precision and normalize
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

bool GaussianFilter::file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

void GaussianFilter::visualizeData(const std::vector<std::vector<std::pair<double, double>>>& data) {
    CTikz test;
    test.start_axis();

    // Set colors for each data series
    std::vector<std::string> colors = {"red", "blue", "green", "pink"};

    // Plot each data series with a different color
    for (size_t i = 0; i < data.size(); ++i) {
        const std::vector<std::pair<double, double>>& seriesData = data[i];
        FunctionStyle style;
        style.color = colors[i];
        test.drawFunc(seriesData, style);
    }

    test.end_axis();
    test.create_tikz_file("result");
}

std::vector<std::vector<std::pair<double, double>>> GaussianFilter::calculateDeviations(const cv::Mat& image, const std::string& mode, double sigma) {
    cv::Mat deviationImage;
    cv::Mat filteredImage;
    cv::Mat filteredImage5;
    cv::Mat filterIm;

    if (mode == "gaussian") {
        filteredImage = gaussian_filter(image, sigma);
        filteredImage5 = gaussian_filter(image, sigma * 5);
    }
    else if (mode == "deriche") {
        cv::GaussianBlur(image, filteredImage,cv::Size(0, 0), sigma);
        filteredImage = filteredImage;
        cv::GaussianBlur(image, filteredImage5,cv::Size(0, 0), sigma * 5);
        filteredImage5 = filteredImage5;
        recursiveGaussianDeriche(image, filterIm, sigma);


    }

    std::vector<std::pair<double, double>> inputCoordinates;
    std::vector<std::pair<double, double>> outputCoordinates;
    std::vector<std::pair<double, double>> coordinatesSigma5;
    std::vector<std::pair<double, double>> differenceCoordinates;

    int centerRow = image.rows / 2;
    int k = (image.cols/2);
    int b = k / 3;
    const double* imageRow = image.ptr<double>(centerRow);
    const double* filteredRow = filteredImage.ptr<double>(centerRow);
    const double* filteredRow5 = filteredImage5.ptr<double>(centerRow);

    for (int x = k - b; x < k + b ; ++x) {
        double inputX = x;
        double inputY = imageRow[x];
        double outputY = filteredRow[x];
        double sigma5Y = filteredRow5[x];
        double difference = std::abs(outputY - sigma5Y);

        inputCoordinates.emplace_back(inputX, inputY);
        outputCoordinates.emplace_back(inputX, outputY);
        coordinatesSigma5.emplace_back(inputX, sigma5Y);
        differenceCoordinates.emplace_back(inputX, difference);
    }

    saveCoordinatesToFile("data1.txt", inputCoordinates);
    saveCoordinatesToFile("data2.txt", outputCoordinates);
    saveCoordinatesToFile("data3.txt", coordinatesSigma5);
    saveCoordinatesToFile("data4.txt", differenceCoordinates);

    visualizeData({ inputCoordinates, outputCoordinates, coordinatesSigma5, differenceCoordinates });

    return { inputCoordinates, outputCoordinates, coordinatesSigma5, differenceCoordinates };
}


void GaussianFilter::saveCoordinatesToFile(const std::string& filename, const std::vector<std::pair<double, double>>& coordinates) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& coord : coordinates) {
            file << coord.first << " " << coord.second << std::endl;
        }
        file.close();
    }
    else {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
    }
}

void GaussianFilter::compare_algorithms(const std::string & input_filename, double sigma, const std::string & algorithm_type, const std::string & visualization, const std::string & output_file_path1) {
    if (!file_exists(input_filename)) {
        std::cerr << "Error: Input file " << input_filename << " does not exist." << std::endl;
        return;
    }
    if (algorithm_type != "gaussian" && algorithm_type != "deriche") {
        std::cerr << "Error: Invalid algorithm type. Must be 'gaussian', 'deriche', ." << std::endl;
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
            } else {
                std::cerr << "Error: Output file path " << output_file_path1 << " already exists." << std::endl;
                return;
            }
        }

        if (visualization == "yes") {
            std::vector<std::vector<std::pair<double, double>>> deviations = calculateDeviations(input, algorithm_type,sigma);
            visualizeData(deviations);
        }
    }

    if (algorithm_type == "deriche") {
        cv::Mat recursive_gaussian_output(input.size(), CV_64F);
        cv::GaussianBlur(input, recursive_gaussian_output,cv::Size(0, 0), sigma);
        if (!output_file_path1.empty()) {
            if (!file_exists(output_file_path1)) {
                cv::Mat output_image;
                recursive_gaussian_output.convertTo(output_image, CV_64F);
                cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
                cv::imwrite(output_file_path1, output_image);
            }
        }

        if (visualization == "yes") {
            std::vector<std::vector<std::pair<double, double>>> deviations = calculateDeviations(input,algorithm_type,sigma);
            visualizeData(deviations);
        }
    }
}

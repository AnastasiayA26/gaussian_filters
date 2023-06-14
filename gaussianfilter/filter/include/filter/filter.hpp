#ifndef GAUSSIANFILTER_FILTERG_HPP
#define GAUSSIANFILTER_FILTERG_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

class GaussianFilter : public cv::Algorithm {
public:
/**
 * @brief Метод применяет рекурсивный гауссово-Дериша фильтр к входному изображению.
 *
 * @param src Входное изображение для фильтрации.
 * @param dst Выходное изображение для сохранения отфильтрованного результата.
 * @param sigma Значение сигмы для фильтра Гаусса.
 */
    void recursiveGaussianDeriche(const cv::Mat &src, cv::Mat &dst, double sigma);

/**
 * @brief Метод считывает входные данные из файла.
 *
 * @param filename Имя файла для чтения данных.
 * @return Входные данные в виде объекта cv::Mat.
 */
    cv::Mat read_input_data(const std::string &filename);

/**
 * @brief Метод применяет гауссов фильтр к входному изображению.
 *
 * @param input Входное изображение для фильтрации.
 * @param sigma Значение сигмы для фильтра Гаусса.
 * @return Отфильтрованное изображение в виде объекта cv::Mat.
 */
    cv::Mat gaussian_filter(const cv::Mat &input, double sigma);

/**
 * @brief Метод сохраняет координаты для визуализации.
 *
 * @param filename Имя файла для сохранения координат.
 * @param coordinates Координаты для сохранения.
 */
    void saveCoordinatesToFile(const std::string &filename, const std::vector<std::pair<double, double>> &coordinates);

/**
 * @brief Метод проверяет, существует ли файл с указанным именем.
 *
 * @param filename Имя файла для проверки.
 * @return true, если файл существует, false в противном случае.
 */
    bool file_exists(const std::string &filename);

/**
 * @brief Метод создает файл result.tex для визуализации.
 *
 * @param data Данные для визуализации.
 */
    void visualizeData(const std::vector<std::vector<std::pair<double, double>>> &data);

/**
 * @brief Метод вычисляет отклонения на основе указанного изображения, алгоритма и значения сигмы.
 *
 * @param image Входное изображение.
 * @param mode Режим алгоритма (например, "gaussian", "deriche").
 * @param sigma Значение сигмы для алгоритма.
 * @return Вектор векторов, содержащий вычисленные отклонения.
 */
    std::vector<std::vector<std::pair<double, double>>>
    calculateDeviations(const cv::Mat &image, const std::string &mode, double sigma);

/**
 * @brief Метод сравнивает различные алгоритмы гауссовой фильтрации путем применения их к входному изображению, визуализирует отклонений при необходимости.
 *
 * @param input_filename Имя входного файла.
 * @param sigma Значение сигмы для фильтра Гаусса.
 * @param algorithm_type Тип алгоритма (например, "gaussian", "deriche").
 * @param visualization Вид визуализации (например, "image", "graph").
 * @param output_file_path1 Путь к файлу для сохранения результата.
 */
    void compare_algorithms(const std::string &input_filename, double sigma, const std::string &algorithm_type,
                            const std::string &visualization, const std::string &output_file_path1);
};
#endif // GAUSSIANFILTER_FILTERG_HPP


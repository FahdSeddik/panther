#include "experiment_utils.h"

#include <fstream>
#include <stdexcept>

/**
 * @brief Logs the results of an experiment to a file.
 *
 * This function appends the results of an experiment, including parameters and metrics,
 * to a specified log file in CSV format.
 *
 * @param filename The name of the log file
 * @param parameters The parameters used in the experiment
 * @param metrics The metrics/results of the experiment
 */
void log_experiment_results(const std::string &filename,
                            const std::map<std::string, double> &parameters,
                            const std::map<std::string, double> &metrics) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;
    for (const auto &p : parameters) file << p.first << "=" << p.second << ",";
    for (const auto &m : metrics) file << m.first << "=" << m.second << ",";
    file << std::endl;
    file.close();
}

/**
 * @brief Runs a parameter sweep for sketched_conv2d_forward and logs results.
 *
 * This function runs sketched_conv2d_forward with different parameter combinations
 * and logs the results for later analysis.
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride_options List of stride options
 * @param padding_options List of padding options
 * @param kernel_size_options List of kernel size options
 * @param log_file Log file to save results
 */
void run_parameter_sweep(const torch::Tensor &x,
                         const torch::Tensor &S1s,
                         const torch::Tensor &U1s,
                         const std::vector<std::vector<int64_t>> &stride_options,
                         const std::vector<std::vector<int64_t>> &padding_options,
                         const std::vector<std::vector<int64_t>> &kernel_size_options,
                         const std::string &log_file) {
    // This is a stub. Actual implementation should call sketched_conv2d_forward and analyze_performance.
}

/**
 * @brief Saves a tensor to a binary file for later analysis.
 *
 * @param filename The name of the file
 * @param tensor The tensor to save
 */
void save_tensor_to_file(const std::string &filename, const torch::Tensor &tensor) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;
    auto data = tensor.contiguous().data_ptr<float>();
    file.write(reinterpret_cast<const char *>(data), tensor.numel() * sizeof(float));
    file.close();
}

/**
 * @brief Loads a tensor from a binary file.
 *
 * @param filename The name of the file
 * @param shape The shape of the tensor
 * @return torch::Tensor The loaded tensor
 */
torch::Tensor load_tensor_from_file(const std::string &filename, const std::vector<int64_t> &shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    std::vector<float> buffer(numel);
    file.read(reinterpret_cast<char *>(buffer.data()), numel * sizeof(float));
    file.close();
    return torch::from_blob(buffer.data(), shape, torch::kFloat32).clone();
}
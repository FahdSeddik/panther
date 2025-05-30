#pragma once
#include <torch/torch.h>

#include <map>
#include <string>
#include <vector>

void log_experiment_results(const std::string &filename,
                            const std::map<std::string, double> &parameters,
                            const std::map<std::string, double> &metrics);

void run_parameter_sweep(const torch::Tensor &x,
                         const torch::Tensor &S1s,
                         const torch::Tensor &U1s,
                         const std::vector<std::vector<int64_t>> &stride_options,
                         const std::vector<std::vector<int64_t>> &padding_options,
                         const std::vector<std::vector<int64_t>> &kernel_size_options,
                         const std::string &log_file);

void save_tensor_to_file(const std::string &filename, const torch::Tensor &tensor);
torch::Tensor load_tensor_from_file(const std::string &filename, const std::vector<int64_t> &shape);
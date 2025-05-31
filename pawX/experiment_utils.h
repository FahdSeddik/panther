#pragma once
#include <torch/torch.h>

#include <map>
#include <string>
#include <vector>

void log_experiment_results(const std::string &filename,
                            const std::map<std::string, double> &parameters,
                            const std::map<std::string, double> &metrics);

void save_tensor_to_file(const std::string &filename, const torch::Tensor &tensor);
torch::Tensor load_tensor_from_file(const std::string &filename, const std::vector<int64_t> &shape);
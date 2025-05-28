#pragma once

#include <torch/extension.h>

#include <ostream>

torch::Tensor scaled_sign_sketch(int64_t m, int64_t n,
                                 c10::optional<torch::Device> device = c10::nullopt,
                                 c10::optional<torch::Dtype> dtype = c10::nullopt);

enum class DistributionFamily {
    Gaussian,
    Uniform
};

enum class Axis {
    Short,
    Long
};

inline std::ostream& operator<<(std::ostream& os, const DistributionFamily& df) {
    switch (df) {
        case DistributionFamily::Gaussian:
            os << "Gaussian";
            break;
        case DistributionFamily::Uniform:
            os << "Uniform";
            break;
        default:
            os << "Unknown";
            break;
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Axis& df) {
    switch (df) {
        case Axis::Short:
            os << "Short";
            break;
        case Axis::Long:
            os << "Long";
            break;
        default:
            os << "Unknown";
            break;
    }
    return os;
}

torch::Tensor dense_sketch_operator(int64_t m, int64_t n,
                                    DistributionFamily distribution,
                                    c10::optional<torch::Device> device = c10::nullopt,
                                    c10::optional<torch::Dtype> dtype = c10::nullopt);

torch::Tensor sparse_sketch_operator(int64_t m, int64_t n, int64_t vec_nnz,
                                     Axis major_axis,
                                     c10::optional<torch::Device> device = c10::nullopt,
                                     c10::optional<torch::Dtype> dtype = c10::nullopt);

std::tuple<torch::Tensor, torch::Tensor> sketch_tensor(const torch::Tensor& input,
                                                       int64_t axis,
                                                       int64_t new_size,
                                                       DistributionFamily distribution,
                                                       c10::optional<torch::Device> device = c10::nullopt,
                                                       c10::optional<torch::Dtype> dtype = c10::nullopt);

torch::Tensor sketch_tensor(const torch::Tensor& input,
                            int64_t axis,
                            int64_t new_size,
                            const torch::Tensor& sketch_matrix,
                            c10::optional<torch::Device> device = c10::nullopt,
                            c10::optional<torch::Dtype> dtype = c10::nullopt);

torch::Tensor gaussian_skop(int64_t m,
                            int64_t d,
                            c10::optional<torch::Device> device = c10::nullopt,
                            c10::optional<torch::Dtype> dtype = c10::nullopt);

torch::Tensor count_skop(int64_t m,
                         int64_t d,
                         c10::optional<torch::Device> device = c10::nullopt,
                         c10::optional<torch::Dtype> dtype = c10::nullopt);

torch::Tensor sjlt_skop(int64_t m,
                        int64_t d,
                        int64_t sparsity = 2,
                        c10::optional<torch::Device> device = c10::nullopt,
                        c10::optional<torch::Dtype> dtype = c10::nullopt);

torch::Tensor srht(torch::Tensor x, int m);

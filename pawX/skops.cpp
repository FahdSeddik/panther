#include "skops.h"

torch::Tensor scaled_sign_sketch(int64_t m, int64_t n, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    // Set tensor options
    auto options = torch::TensorOptions().device(device.value_or(torch::kCPU)).dtype(dtype.value_or(torch::kFloat));

    // Generate {-1, 1} directly using sign function and scale by precomputed factor
    return torch::sign(torch::rand({m, n}, options) - 0.5) * (1.0 / std::sqrt(static_cast<double>(m)));
}

torch::Tensor dense_sketch_operator(int64_t m, int64_t n, DistributionFamily distribution, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    // Set tensor options
    auto options = torch::TensorOptions().device(device.value_or(torch::kCPU)).dtype(dtype.value_or(torch::kFloat));

    // Generate the sketch matrix based on the specified distribution
    if (distribution == DistributionFamily::Gaussian) {
        return torch::randn({m, n}, options);
    } else if (distribution == DistributionFamily::Uniform) {
        return (torch::rand({m, n}, options) - 0.5) * std::sqrt(static_cast<double>(12.0));
    } else {
        throw std::invalid_argument("Unsupported distribution family");
    }
}

// The sketch_tensor function:
// 1. Validates the axis and new_size constraints.
// 2. Generates a sketching operator S of size (new_size x original_size) along the chosen axis.
// 3. Permutes the input tensor so that the axis to sketch becomes the first dimension.
// 4. Flattens the remaining dimensions, applies S * input, and then reshapes and permutes back.
std::tuple<torch::Tensor, torch::Tensor> sketch_tensor(const torch::Tensor &input,
                                                       int64_t axis,
                                                       int64_t new_size,
                                                       DistributionFamily distribution,
                                                       c10::optional<torch::Device> device,
                                                       c10::optional<torch::Dtype> dtype) {
    // Check that axis is valid.
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "Axis index out of bounds");

    // Get size along the axis.
    int64_t original_size = input.size(axis);
    TORCH_CHECK(new_size > 0 && new_size <= original_size,
                "new_size must be > 0 and <= the size of the dimension to sketch");

    // Create the sketching operator: dimensions (new_size x original_size)
    torch::Tensor sketch_matrix = dense_sketch_operator(new_size, original_size, distribution, device, dtype);

    return std::make_tuple(sketch_tensor(input, axis, new_size, sketch_matrix, device, dtype), sketch_matrix);
}

torch::Tensor sketch_tensor(const torch::Tensor &input,
                            int64_t axis,
                            int64_t new_size,
                            const torch::Tensor &sketch_matrix,
                            c10::optional<torch::Device> device,
                            c10::optional<torch::Dtype> dtype) {
    // Check that axis is valid.
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "Axis index out of bounds");

    // Get size along the axis.
    int64_t original_size = input.size(axis);
    TORCH_CHECK(new_size > 0 && new_size <= original_size,
                "new_size must be > 0 and <= the size of the dimension to sketch");

    // Build permutation vector to bring the axis to be sketched to the front.
    std::vector<int64_t> perm;
    perm.push_back(axis);
    for (int64_t i = 0; i < input.dim(); i++) {
        if (i != axis) {
            perm.push_back(i);
        }
    }
    // Permute the input tensor accordingly.
    torch::Tensor input_permuted = input.permute(perm);

    // Collapse all dimensions except the first into a single dimension.
    torch::Tensor reshaped = input_permuted.reshape({original_size, -1});

    // Apply the sketching operator.
    // S is (new_size x original_size) and input reshaped is (original_size x rest)
    torch::Tensor output_flat = torch::matmul(sketch_matrix, reshaped);

    // Determine the new shape:
    // Replace the sketched axis with new_size while keeping all other dimensions the same.
    std::vector<int64_t> rest_dims;
    for (int64_t i = 0; i < input.dim(); i++) {
        if (i != axis) {
            rest_dims.push_back(input.size(i));
        }
    }
    std::vector<int64_t> new_shape;
    new_shape.push_back(new_size);
    new_shape.insert(new_shape.end(), rest_dims.begin(), rest_dims.end());

    // Reshape the result back to the permuted shape.
    torch::Tensor output_permuted = output_flat.reshape(new_shape);

    // Compute inverse permutation to restore original ordering.
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        inv_perm[perm[i]] = i;
    }
    torch::Tensor output = output_permuted.permute(inv_perm);

    return output;
}
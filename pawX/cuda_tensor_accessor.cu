#include "cuda_tensor_accessor.cuh"
#include "linear.h"

void test_tensor_accessor(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 2, "Expected 2 dimensions, got ", tensor.dim());
    std::cout << "Original tensor:\n"
              << tensor << std::endl;

    // Create a non-contiguous tensor by transposing
    auto non_contiguous = tensor.t();
    std::cout << "Non-contiguous tensor:\n"
              << non_contiguous << std::endl;

    if (non_contiguous.scalar_type() == at::ScalarType::Half) {
        const FlexibleTensorAccessor<c10::Half, 2>& accessor = tensor_utils::buildAccessor<c10::Half, 2>(non_contiguous);
        std::cout << "Accessing elements through accessor:" << std::endl;
        for (int i = 0; i < accessor.size(0); ++i) {
            for (int j = 0; j < accessor.size(1); ++j) {
                printf("accessor(%d,%d) = %f\n", i, j, __half2float(accessor(i, j)));
            }
        }
        std::cout << "Accessor sizes: " << accessor.size(0) << ", " << accessor.size(1) << std::endl;
        std::cout << "Accessor strides: " << accessor.stride(0) << ", " << accessor.stride(1) << std::endl;
    } else {
        const FlexibleTensorAccessor<float, 2>& accessor = tensor_utils::buildAccessor<float, 2>(non_contiguous);
        std::cout << "Accessing elements through accessor:" << std::endl;
        for (int i = 0; i < accessor.size(0); ++i) {
            for (int j = 0; j < accessor.size(1); ++j) {
                printf("accessor(%d,%d) = %f\n", i, j, accessor(i, j));
            }
        }
        std::cout << "Accessor sizes: " << accessor.size(0) << ", " << accessor.size(1) << std::endl;
        std::cout << "Accessor strides: " << accessor.stride(0) << ", " << accessor.stride(1) << std::endl;
    }
}
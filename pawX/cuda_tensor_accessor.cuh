#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Forward declaration for type traits
template <typename T, typename Enable = void>
struct is_valid_tensor_type : std::false_type {};

// Base specializations for supported types
template <>
struct is_valid_tensor_type<float> : std::true_type {};

template <>
struct is_valid_tensor_type<double> : std::true_type {};

// Add CUDA half-precision support
template <>
struct is_valid_tensor_type<__half> : std::true_type {};

// Type mapping from PyTorch to CUDA types
template <typename T>
struct cuda_type {
    using type = T;
};

template <>
struct cuda_type<c10::Half> {
    using type = __half;
    __host__ __device__ inline const static __half get_zero() { return __float2half(0.0f); }
};

template <>
struct cuda_type<float> {
    using type = float;
    __host__ __device__ inline const static float get_zero() { return 0.0f; }
};

template <typename T>
using cuda_type_t = typename cuda_type<T>::type;

// CUDA-specific accessor that adds device qualifiers
template <typename T, int N>
struct FlexibleTensorAccessor {
    static_assert(is_valid_tensor_type<cuda_type_t<T>>::value, "Unsupported tensor type");
    cuda_type_t<T>* data;
    int64_t sizes[N];
    int64_t strides[N];

    __host__ __device__ FlexibleTensorAccessor() : data(nullptr) {
        for (int i = 0; i < N; ++i) {
            sizes[i] = 0;
            strides[i] = 0;
        }
    }

    __host__ __device__ FlexibleTensorAccessor(cuda_type_t<T>* data_, const int64_t* sizes_, const int64_t* strides_)
        : data(data_) {
        for (int i = 0; i < N; ++i) {
            sizes[i] = sizes_[i];
            strides[i] = strides_[i];
        }
    }

    // Override methods with CUDA qualifiers
    __host__ __device__ inline int64_t numel() const {
        int64_t total = 1;
        for (int i = 0; i < N; ++i) {
            total *= sizes[i];
        }
        return total;
    }

    __host__ __device__ inline int64_t size(int dim) const {
        return sizes[dim];
    }

    __host__ __device__ inline int64_t stride(int dim) const {
        return strides[dim];
    }

    __host__ __device__ inline int64_t offset(const int64_t idx[N]) const {
        int64_t off = 0;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            off += idx[i] * strides[i];
        }
        return off;
    }

    // Read as a specified type (e.g., float, double, __half, etc.)
    template <typename OutType = cuda_type_t<T>, typename... Args>
    __host__ __device__ inline OutType get(Args... args) const {
        static_assert(sizeof...(args) == N, "Invalid number of indices");
        int64_t idx[N] = {static_cast<int64_t>(args)...};
        auto& val = data[offset(idx)];
        if constexpr (std::is_same_v<cuda_type_t<T>, __half> && std::is_same_v<OutType, float>) {
            return __half2float(val);
        } else if constexpr (std::is_same_v<cuda_type_t<T>, __half> && std::is_same_v<OutType, double>) {
            return static_cast<double>(__half2float(val));
        } else if constexpr (std::is_same_v<cuda_type_t<T>, float> && std::is_same_v<OutType, __half>) {
            return __float2half(val);
        } else if constexpr (std::is_same_v<cuda_type_t<T>, double> && std::is_same_v<OutType, __half>) {
            return __float2half(static_cast<float>(val));
        } else {
            return static_cast<OutType>(val);
        }
    }

    // Write as a specified type (e.g., float, double, __half, etc.)
    template <typename InType = cuda_type_t<T>, typename... Args>
    __host__ __device__ inline void set(InType value, Args... args) {
        static_assert(sizeof...(args) == N, "Invalid number of indices");
        int64_t idx[N] = {static_cast<int64_t>(args)...};
        if constexpr (std::is_same_v<cuda_type_t<T>, __half> && std::is_same_v<InType, float>) {
            data[offset(idx)] = __float2half(value);
        } else if constexpr (std::is_same_v<cuda_type_t<T>, __half> && std::is_same_v<InType, double>) {
            data[offset(idx)] = __float2half(static_cast<float>(value));
        } else if constexpr (std::is_same_v<cuda_type_t<T>, float> && std::is_same_v<InType, __half>) {
            data[offset(idx)] = __half2float(value);
        } else if constexpr (std::is_same_v<cuda_type_t<T>, double> && std::is_same_v<InType, __half>) {
            data[offset(idx)] = static_cast<double>(__half2float(value));
        } else {
            data[offset(idx)] = static_cast<cuda_type_t<T>>(value);
        }
    }

    // Original operator() for direct access (no conversion)
    template <typename... Args>
    __host__ __device__ inline cuda_type_t<T>& operator()(Args... args) {
        static_assert(sizeof...(args) == N, "Invalid number of indices");
        int64_t idx[N] = {static_cast<int64_t>(args)...};
        return data[offset(idx)];
    }

    template <typename... Args>
    __host__ __device__ inline const cuda_type_t<T>& operator()(Args... args) const {
        static_assert(sizeof...(args) == N, "Invalid number of indices");
        int64_t idx[N] = {static_cast<int64_t>(args)...};
        return data[offset(idx)];
    }

    __host__ __device__ inline cuda_type_t<T>* get_data() const {
        return data;
    }
};

namespace tensor_utils {

template <typename T, int N>
inline FlexibleTensorAccessor<T, N> buildAccessor(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == N, "Expected ", N, " dimensions, got ", tensor.dim());

    // Check tensor type
    if constexpr (std::is_same_v<T, float>) {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float,
                    "Expected Float32 tensor, got ", tensor.scalar_type());
    } else if constexpr (std::is_same_v<T, double>) {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Double,
                    "Expected Double tensor, got ", tensor.scalar_type());
    } else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, c10::Half>) {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Half,
                    "Expected Float16 tensor, got ", tensor.scalar_type());
    }

    // Get tensor properties
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();

    // Convert sizes and strides to int64_t arrays
    int64_t sizes_array[N];
    int64_t strides_array[N];

    for (int i = 0; i < N; ++i) {
        sizes_array[i] = sizes[i];
        strides_array[i] = strides[i];
    }

    // Handle data pointer based on type
    cuda_type_t<T>* data_ptr;
    if constexpr (std::is_same_v<T, float>) {
        data_ptr = tensor.data_ptr<float>();
    } else if constexpr (std::is_same_v<T, double>) {
        data_ptr = tensor.data_ptr<double>();
    } else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, c10::Half>) {
        auto* raw_ptr = tensor.data_ptr<c10::Half>();
        data_ptr = reinterpret_cast<__half*>(raw_ptr);
    }

    return FlexibleTensorAccessor<T, N>(data_ptr, sizes_array, strides_array);
}

// instantiate for different dimensions
#define INSTANTIATE_GET_ACCESSOR(N)                                                                   \
    template FlexibleTensorAccessor<float, N> buildAccessor<float, N>(const torch::Tensor& tensor);   \
    template FlexibleTensorAccessor<double, N> buildAccessor<double, N>(const torch::Tensor& tensor); \
    template FlexibleTensorAccessor<__half, N> buildAccessor<__half, N>(const torch::Tensor& tensor); \
    template FlexibleTensorAccessor<c10::Half, N> buildAccessor<c10::Half, N>(const torch::Tensor& tensor)

INSTANTIATE_GET_ACCESSOR(1);
INSTANTIATE_GET_ACCESSOR(2);
INSTANTIATE_GET_ACCESSOR(3);
INSTANTIATE_GET_ACCESSOR(4);
INSTANTIATE_GET_ACCESSOR(5);
INSTANTIATE_GET_ACCESSOR(6);

}  // namespace tensor_utils

// Test function to verify tensor accessor functionality
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
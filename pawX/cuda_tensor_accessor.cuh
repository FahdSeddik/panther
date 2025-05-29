#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T, typename Enable = void>
struct is_valid_tensor_type : std::false_type {};

template <>
struct is_valid_tensor_type<float> : std::true_type {};

template <>
struct is_valid_tensor_type<double> : std::true_type {};

template <>
struct is_valid_tensor_type<__half> : std::true_type {};

// Type mapping from PyTorch to CUDA types
template <typename T>
struct cuda_type {
    using type = T;
};

/**
 * @brief Specialization of the cuda_type struct for c10::Half type.
 *
 * This specialization maps the c10::Half type to CUDA's native __half type,
 * and provides utility functions for zero initialization and addition.
 *
 * - type: Alias for CUDA's __half type.
 * - get_zero(): Returns a __half value representing zero.
 * - add(a, b): Returns the sum of two __half values using CUDA's half-precision addition.
 */
template <>
struct cuda_type<c10::Half> {
    using type = __half;
    __host__ __device__ inline static __half get_zero() { return __float2half(0.0f); }
    __host__ __device__ inline static __half add(__half a, __half b) { return __hadd(a, b); }
};

/**
 * @brief Specialization of the cuda_type struct for float type.
 *
 * This specialization defines the CUDA-compatible type for float,
 * and provides utility functions for common operations:
 * - get_zero(): Returns the additive identity (0.0f) for float.
 * - add(a, b): Returns the sum of two float values.
 *
 * All member functions are marked with __host__ __device__ to allow usage
 * in both host and device code.
 */
template <>
struct cuda_type<float> {
    using type = float;
    __host__ __device__ inline static float get_zero() { return 0.0f; }
    __host__ __device__ inline static float add(float a, float b) { return a + b; }
};

template <typename T>
using cuda_type_t = typename cuda_type<T>::type;

// CUDA-specific accessor that adds device qualifiers
/**
 * @brief A flexible N-dimensional tensor accessor for CUDA device and host code.
 *
 * This struct provides a generic interface for accessing and manipulating tensor data
 * of arbitrary type and dimensionality on both host and device. It supports type conversion
 * between common CUDA types (e.g., float, double, __half) during read and write operations.
 *
 * @tparam T The logical element type of the tensor (e.g., float, double, at::Half, etc.).
 * @tparam N The number of dimensions of the tensor.
 *
 * Members:
 * - data: Pointer to the underlying tensor data (device or host memory).
 * - sizes: Array of dimension sizes.
 * - strides: Array of strides for each dimension.
 *
 * Constructors:
 * - Default constructor initializes data to nullptr and sizes/strides to zero.
 * - Parameterized constructor initializes data, sizes, and strides from arguments.
 *
 * Methods:
 * - numel(): Returns the total number of elements in the tensor.
 * - size(dim): Returns the size of the specified dimension.
 * - stride(dim): Returns the stride of the specified dimension.
 * - offset(idx): Computes the linear offset for a given N-dimensional index.
 * - get<OutType>(...): Reads an element at the specified indices, converting to OutType if needed.
 * - set<InType>(value, ...): Writes a value at the specified indices, converting from InType if needed.
 * - operator()(...): Direct access to the underlying element at the specified indices.
 * - get_data(): Returns the raw data pointer.
 *
 * Notes:
 * - All methods are marked with __host__ __device__ for use in both host and device code.
 * - Type conversions between float, double, and __half are handled automatically in get/set.
 * - Static assertions ensure correct usage (e.g., number of indices matches N).
 *
 * Example usage:
 * @code
 * FlexibleTensorAccessor<float, 2> acc(data_ptr, sizes, strides);
 * float val = acc.get(0, 1); // Read element at (0, 1)
 * acc.set(3.14f, 0, 1);      // Write value at (0, 1)
 * @endcode
 */
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
    /**
     * @brief Retrieves the tensor element at the specified indices, with optional type conversion.
     *
     * This function accesses the element at the position specified by the variadic indices `args`.
     * The number of indices must match the tensor's rank `N`. The return type can be specified via
     * the template parameter `OutType`, defaulting to the CUDA-compatible type of `T`.
     *
     * If the underlying storage type and the requested output type differ (e.g., `__half` to `float`),
     * appropriate CUDA conversion functions are used to perform the cast.
     *
     * @tparam OutType The desired output type (defaults to `cuda_type_t<T>`).
     * @tparam Args    The types of the indices (should be convertible to `int64_t`).
     * @param args     The indices specifying the position of the element to retrieve.
     * @return         The value at the specified indices, converted to `OutType` if necessary.
     *
     * @note This function is both __host__ and __device__ callable.
     * @throws static_assert if the number of indices does not match the tensor's rank.
     */
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
    /**
     * @brief Sets the value at the specified multi-dimensional index in the tensor.
     *
     * This function assigns the given value to the tensor element located at the position
     * specified by the provided indices. It handles type conversions between different
     * floating-point types, including `float`, `double`, and CUDA's `__half` type, ensuring
     * correct casting and conversion as needed.
     *
     * @tparam InType The input value type (defaults to the CUDA-compatible type for T).
     * @tparam Args   Variadic template parameter pack representing the index types.
     * @param value   The value to set at the specified index.
     * @param args    The indices specifying the position in the tensor (must match tensor rank N).
     *
     * @note
     * - The number of indices provided in `args` must match the tensor's rank (N).
     * - Performs appropriate type conversion for half-precision and floating-point types.
     * - Can be called from both host and device code.
     *
     * @throws static_assert if the number of indices does not match the tensor's rank.
     */
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

/**
 * @brief Builds a FlexibleTensorAccessor for a given torch::Tensor.
 *
 * This function checks that the input tensor matches the expected number of dimensions (N)
 * and the expected scalar type (float, double, or half). It then extracts the tensor's
 * sizes and strides, converts them to fixed-size int64_t arrays, and obtains a properly
 * typed data pointer for CUDA access. The function finally constructs and returns a
 * FlexibleTensorAccessor<T, N> for efficient device-side tensor access.
 *
 * @tparam T The scalar type of the tensor (float, double, __half, or c10::Half).
 * @tparam N The number of dimensions expected in the tensor.
 * @param tensor The input torch::Tensor to be accessed.
 * @return FlexibleTensorAccessor<T, N> An accessor for the tensor data, sizes, and strides.
 *
 * @throws c10::Error if the tensor does not have the expected number of dimensions or type.
 */
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

    auto sizes = tensor.sizes();
    auto strides = tensor.strides();

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

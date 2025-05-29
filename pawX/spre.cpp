#include "spre.h"

sinSRPEImpl::sinSRPEImpl(int64_t num_heads_,
                             int64_t perHead_in_,
                             int64_t sines_,
                             int64_t num_realizations_,
                             torch::Device device_,
                             torch::Dtype dtype_)
    
    : num_heads(num_heads_),
        perHead_in(perHead_in_),
        sines(sines_),
        num_realizations(num_realizations_),
        device(device_),
        dtype(dtype_) {
    auto options = torch::TensorOptions().device(device).dtype(dtype);

    // Register parameters
    freqs = register_parameter(
        "freqs",
        torch::randn({num_heads, perHead_in, sines}, options)
    );

    phases = register_parameter(
        "phases",
        torch::randn({num_heads, perHead_in, sines}, options)
    );

    scales = register_parameter(
        "scales",
        torch::randn({num_heads, perHead_in, sines}, options)
    );

    z = register_parameter(
        "z",
        torch::randn({1, num_heads, perHead_in, 2 * sines, num_realizations}, options)
    );

    
    // In-place normalization of scales
    {
        torch::NoGradGuard no_grad;
        // auto norm_vals = scales.norm(-1, /*keepdim=*/true);
        auto norm_vals = torch::norm(scales, 2, /*dim=*/-1, /*keepdim=*/true);
        // print norm_vals size and scales size
        std::cout << "norm_vals size: " << norm_vals.sizes() << std::endl;
        std::cout << "scales size: " << scales.sizes() << std::endl;
    
        scales.div_(norm_vals / 2.0);
    }

    // Shift freqs by constant
    freqs.data().sub_(4.0);

}

std::pair<torch::Tensor, torch::Tensor> sinSRPEImpl::forward(int64_t len)  {
    auto options = torch::TensorOptions().device(device).dtype(dtype);

    // Build indices: shape (1,1,len,1)
    torch::Tensor indices = torch::arange(0, len, options)
    .unsqueeze(0).unsqueeze(0).unsqueeze(-1);

    // Sigmoid and scale freqs: shape (num_heads, perHead_in, 1, sines)
    torch::Tensor f = torch::sigmoid(freqs) / 2.0;
    f = f.unsqueeze(-2);

    // Phases for Q
    torch::Tensor phases_q = 2 * M_PI * indices * f + phases.unsqueeze(-2);
    torch::Tensor omega_q = torch::cat({
        torch::cos(phases_q),
        torch::sin(phases_q)
    }, -1)
    .unsqueeze(0);

    // Phases for K
    torch::Tensor phases_k = 2 * M_PI * indices * f;
    torch::Tensor omega_k = torch::cat({
        torch::cos(phases_k),
        torch::sin(phases_k)
    }, -1)
    .unsqueeze(0);

    // Softplus scales and apply
    torch::Tensor sc = torch::nn::functional::softplus(scales);
    sc = torch::cat({sc, sc}, -1); // Duplicate for cos and sin
    sc = sc.unsqueeze(0).unsqueeze(-1);
    torch::Tensor z_scaled = z * sc;

    // Compute qbar and kbar
    auto qbar = torch::matmul(omega_q, z_scaled);
    auto kbar = torch::matmul(omega_k, z_scaled);

    // Permute dimensions to match Python output
    qbar = qbar.permute({0, 3, 1, 2, 4});
    kbar = kbar.permute({0, 3, 1, 2, 4});

    // Final scaling
    double scale_val = std::pow(num_realizations * perHead_in, 0.25);
    qbar = qbar / scale_val;
    kbar = kbar / scale_val;

    return {qbar, kbar};
}
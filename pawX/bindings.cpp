#include "attention.h"
#include "conv2d.h"
#include "cqrrpt.h"
#include "linear.h"
#include "rsvd.h"
#include "skops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_sign_sketch", &scaled_sign_sketch,
          py::arg("m"), py::arg("n"),
          py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("sketched_linear_forward", &sketched_linear_forward,
          "Sketched Linear Forward Pass",
          py::arg("input"), py::arg("S1s"), py::arg("S2s"),
          py::arg("U1s"), py::arg("U2s"), py::arg("bias"),
          py::arg("use_tensor_core") = false);

    m.def("sketched_linear_backward", &sketched_linear_backward,
          "Sketched Linear Backward Pass",
          py::arg("grad_output"), py::arg("input"), py::arg("S1s"),
          py::arg("S2s"), py::arg("U1s"), py::arg("U2s"), py::arg("use_tensor_core") = false);

    py::enum_<DistributionFamily>(m, "DistributionFamily")
        .value("Gaussian", DistributionFamily::Gaussian)
        .value("Uniform", DistributionFamily::Uniform)
        .export_values();

    m.def("cqrrpt", &cqrrpt, py::arg("M"), py::arg("gamma") = 1.25, py::arg("F") = DistributionFamily::Gaussian);
    m.def("randomized_svd", &randomized_svd, py::arg("A"), py::arg("k"), py::arg("tol"));

    m.def("dense_sketch_operator", &dense_sketch_operator,
          py::arg("m"),
          py::arg("n"),
          py::arg("distribution"),
          py::arg("device") = py::none(),
          py::arg("dtype") = py::none());

    m.def("sketch_tensor",
          py::overload_cast<const torch::Tensor&, int64_t, int64_t, DistributionFamily, c10::optional<torch::Device>, c10::optional<torch::Dtype>>(&sketch_tensor),
          "Sketch Tensor with Distribution",
          py::arg("input"), py::arg("axis"), py::arg("new_size"), py::arg("distribution"),
          py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);

    m.def("sketch_tensor",
          py::overload_cast<const torch::Tensor&, int64_t, int64_t, const torch::Tensor&, c10::optional<torch::Device>, c10::optional<torch::Dtype>>(&sketch_tensor),
          "Sketch Tensor with Sketch Matrix",
          py::arg("input"), py::arg("axis"), py::arg("new_size"), py::arg("sketch_matrix"),
          py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);

    m.def("causal_numerator_apply", &causal_numerator_apply,
          py::arg("query_prime"), py::arg("key_prime"), py::arg("value_prime"));

    m.def("causal_denominator_apply", &causal_denominator_apply,
          py::arg("query_prime"), py::arg("key_prime"));

    m.def("rmha_forward", &rmha_forward,
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("Wq"), py::arg("Wk"), py::arg("Wv"), py::arg("W0"),
          py::arg("num_heads"), py::arg("embed_dim"), py::arg("kernel_fn"),
          py::arg("causal"),
          py::arg("attention_mask") = c10::nullopt,
          py::arg("bq") = c10::nullopt, py::arg("bk") = c10::nullopt,
          py::arg("bv") = c10::nullopt, py::arg("b0") = c10::nullopt,
          py::arg("projection_matrix") = c10::nullopt);

    m.def("create_projection_matrix", &create_projection_matrix,
          py::arg("m"), py::arg("d"), py::arg("seed") = 42, py::arg("scaling") = false,
          py::arg("dtype") = c10::nullopt, py::arg("device") = c10::nullopt);

    m.def("sketched_conv2d_forward", &sketched_conv2d_forward,
          py::arg("x"), py::arg("S1s"),
          py::arg("U1s"), py::arg("stride"),
          py::arg("padding"), py::arg("kernel_size"), py::arg("bias"));

    m.def("sketched_conv2d_backward", &sketched_conv2d_backward,
          py::arg("input"), py::arg("S1s"), py::arg("S2s"),
          py::arg("U1s"), py::arg("U2s"), py::arg("stride"),
          py::arg("padding"), py::arg("kernel_size"), py::arg("in_shape"),
          py::arg("grad_out"));
}

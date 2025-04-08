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
          py::arg("U1s"), py::arg("U2s"), py::arg("bias"));

    m.def("sketched_linear_backward", &sketched_linear_backward,
          "Sketched Linear Backward Pass",
          py::arg("grad_output"), py::arg("input"), py::arg("S1s"),
          py::arg("S2s"), py::arg("U1s"), py::arg("U2s"));

    m.def("cqrrpt", &cqrrpt, py::arg("M"), py::arg("gamma") = 1.25, py::arg("F") = "default");
    m.def("randomized_svd", &randomized_svd, py::arg("A"), py::arg("k"), py::arg("tol"));

    py::enum_<DistributionFamily>(m, "DistributionFamily")
        .value("Gaussian", DistributionFamily::Gaussian)
        .value("Uniform", DistributionFamily::Uniform)
        .export_values();

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
}

#include "linear.h"
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
}

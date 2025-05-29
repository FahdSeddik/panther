#include "attention.h"
#include "conv2d.h"
#include "cqrrpt.h"
#include "linear.h"
#include "rsvd.h"
#include "skops.h"
#include "spre.h"
#include <torch/extension.h>
// PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      py::module::import("torch");
      py::module::import("torch.nn");

      m.def("scaled_sign_sketch", &scaled_sign_sketch,
            py::arg("m"), py::arg("n"),
            py::arg("device") = py::none(), py::arg("dtype") = py::none());

      m.def("sketched_linear_forward", &sketched_linear_forward,
            "Sketched Linear Forward Pass",
            py::arg("input"), py::arg("S1s"), py::arg("S2s"),
            py::arg("U1s"), py::arg("U2s"), py::arg("bias") = c10::nullopt,
            py::arg("use_gpu") = false);

      m.def("sketched_linear_backward", &sketched_linear_backward,
            "Sketched Linear Backward Pass",
            py::arg("grad_output"), py::arg("input"), py::arg("S1s"),
            py::arg("S2s"), py::arg("U1s"), py::arg("U2s"), py::arg("has_bias") = false, py::arg("use_gpu") = false);

      py::enum_<DistributionFamily>(m, "DistributionFamily")
          .value("Gaussian", DistributionFamily::Gaussian)
          .value("Uniform", DistributionFamily::Uniform)
          .export_values();
      py::enum_<Axis>(m, "Axis")
          .value("Short", Axis::Short)
          .value("Long", Axis::Long)
          .export_values();

      m.def("cqrrpt", &cqrrpt, py::arg("M"), py::arg("gamma") = 1.25, py::arg("F") = DistributionFamily::Gaussian);
      m.def("randomized_svd", &randomized_svd, py::arg("A"), py::arg("k"), py::arg("tol"));

      m.def("dense_sketch_operator", &dense_sketch_operator,
            py::arg("m"),
            py::arg("n"),
            py::arg("distribution"),
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none());

      m.def("sparse_sketch_operator", &sparse_sketch_operator,
            py::arg("m"),
            py::arg("n"),
            py::arg("vec_nnz"),
            py::arg("major_axis"),
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none());

      m.def("sketch_tensor",
            py::overload_cast<const torch::Tensor &, int64_t, int64_t, DistributionFamily, c10::optional<torch::Device>, c10::optional<torch::Dtype>>(&sketch_tensor),
            "Sketch Tensor with Distribution",
            py::arg("input"), py::arg("axis"), py::arg("new_size"), py::arg("distribution"),
            py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);

      m.def("sketch_tensor",
            py::overload_cast<const torch::Tensor &, int64_t, int64_t, const torch::Tensor &, c10::optional<torch::Device>, c10::optional<torch::Dtype>>(&sketch_tensor),
            "Sketch Tensor with Sketch Matrix",
            py::arg("input"), py::arg("axis"), py::arg("new_size"), py::arg("sketch_matrix"),
            py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);

      m.def("causal_numerator_forward", &causal_numerator_forward,
            py::arg("qs"), py::arg("ks"), py::arg("vs"));

      m.def("causal_numerator_backward", &causal_numerator_backward,
            py::arg("res_grad"), py::arg("sums"), py::arg("qs"),
            py::arg("ks"), py::arg("vs"));

      m.def("causal_denominator_forward", &causal_denominator_forward,
            py::arg("qs"), py::arg("ks"));

      m.def("causal_denominator_backward", &causal_denominator_backward,
            py::arg("res_grad"), py::arg("sums"), py::arg("qs"));

      m.def("rmha_forward", &rmha_forward,
            py::arg("query"), py::arg("key"), py::arg("value"),
            py::arg("Wq"), py::arg("Wk"), py::arg("Wv"), py::arg("W0"),
            py::arg("num_heads"), py::arg("embed_dim"), py::arg("kernel_fn"),
            py::arg("causal"),
            py::arg("attention_mask") = c10::nullopt,
            py::arg("bq") = c10::nullopt, py::arg("bk") = c10::nullopt,
            py::arg("bv") = c10::nullopt, py::arg("b0") = c10::nullopt,
            py::arg("projection_matrix") = c10::nullopt,
            py::arg("spre_model") = nullptr);

      m.def("create_projection_matrix", &create_projection_matrix,
            py::arg("m"), py::arg("d"), py::arg("seed") = 42, py::arg("scaling") = false,
            py::arg("dtype") = c10::nullopt, py::arg("device") = c10::nullopt);

      m.def("sketched_conv2d_forward", &sketched_conv2d_forward,
            py::arg("x"), py::arg("S1s"),
            py::arg("U1s"), py::arg("stride"),
            py::arg("padding"), py::arg("kernel_size"), py::arg("bias") = c10::nullopt);

      m.def("sketched_conv2d_backward", &sketched_conv2d_backward,
            py::arg("input"), py::arg("S1s"),
            py::arg("U1s"), py::arg("stride"),
            py::arg("padding"), py::arg("kernel_size"), py::arg("in_shape"),
            py::arg("grad_out"));

      m.def("test_tensor_accessor", &test_tensor_accessor,
            py::arg("tensor"));

      m.def("gaussian_skop", &gaussian_skop,
            py::arg("m"), py::arg("d"),
            py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);
      m.def("count_skop", &count_skop,
            py::arg("m"), py::arg("d"),
            py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);
      m.def("sjlt_skop", &sjlt_skop,
            py::arg("m"), py::arg("d"), py::arg("sparsity") = 2,
            py::arg("device") = c10::nullopt, py::arg("dtype") = c10::nullopt);
      m.def("srht", &srht,
            py::arg("x"), py::arg("m"));

      py::class_<torch::nn::Module, std::shared_ptr<torch::nn::Module>>(m, "Module");

      py::class_<sinSRPEImpl,       /* C++ type */
                 torch::nn::Module, /* base class */
                 std::shared_ptr<sinSRPEImpl> /* holder */>(m, "sinSRPE")
          // Register the parameters
          .def_readwrite("freqs", &sinSRPEImpl::freqs)
          .def_readwrite("phases", &sinSRPEImpl::phases)
          .def_readwrite("scales", &sinSRPEImpl::scales)
          .def_readwrite("z", &sinSRPEImpl::z)
          .def_readwrite("num_realizations", &sinSRPEImpl::num_realizations)
          .def_readwrite("num_heads", &sinSRPEImpl::num_heads)
          .def_readwrite("perHead_in", &sinSRPEImpl::perHead_in)
          .def_readwrite("sines", &sinSRPEImpl::sines)
          .def_readwrite("device", &sinSRPEImpl::device)
          .def_readwrite("dtype", &sinSRPEImpl::dtype)

          .def(
              // this init takes 3 required args + 3 optional ones
              py::init(
                  [](int64_t num_heads,
                     int64_t perHead_in,
                     int64_t sines,
                     int64_t num_realizations,
                     c10::optional<torch::Device> maybe_dev,
                     c10::optional<torch::Dtype> maybe_dt)
                  {
                        auto dev = maybe_dev.value_or(torch::kCPU);
                        auto dt = maybe_dt.value_or(torch::kFloat);
                        return std::make_shared<sinSRPEImpl>(
                            num_heads,
                            perHead_in,
                            sines,
                            num_realizations,
                            dev,
                            dt);
                  }),
              py::arg("num_heads"),
              py::arg("perHead_in"),
              py::arg("sines"),
              py::arg("num_realizations") = 256,
              py::arg("device") = c10::nullopt,
              py::arg("dtype") = c10::nullopt)

          .def("forward", &sinSRPEImpl::forward);
}

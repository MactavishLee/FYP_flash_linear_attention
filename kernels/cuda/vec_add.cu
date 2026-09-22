// Starter CUDA kernel. Called from python/cuda_ops.py.
// Rewrite this yourself in week 1; do not treat it as finished work.

#include <torch/extension.h>

__global__ void vec_add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}

torch::Tensor vec_add_cuda(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "shape mismatch");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "vec_add starter is fp32 only");

    auto out = torch::empty_like(x);
    const int n = static_cast<int>(x.numel());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add_cuda, "fp32 vector add (CUDA)");
}

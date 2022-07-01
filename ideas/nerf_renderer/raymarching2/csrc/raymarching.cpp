#include <torch/extension.h>
#include <cstdint>
#include <vector>

std::vector<torch::Tensor> generate_training_samples(
    torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor aabb,
    torch::Tensor density_bitfield, int max_samples
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generate_training_samples", &generate_training_samples, "");
}
#include <torch/extension.h>
#include <cstdint>
#include <vector>

std::vector<torch::Tensor> generate_training_samples(
    torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor aabb,
    torch::Tensor density_bitfield, int max_samples
);

std::vector<torch::Tensor> volumetric_rendering(
    torch::Tensor rays_o, torch::Tensor indices, torch::Tensor positions, 
    torch::Tensor deltas, torch::Tensor ts, 
    torch::Tensor sigmas, torch::Tensor rgbs, torch::Tensor bkgd_rgb
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generate_training_samples", &generate_training_samples, "");
    m.def("volumetric_rendering", &volumetric_rendering, "");
}
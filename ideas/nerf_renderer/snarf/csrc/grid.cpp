#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "common.h"

std::vector<torch::Tensor> root_finding(
    const torch::Tensor x_init,	const torch::Tensor x_jac
);

std::vector<torch::Tensor> grid_sample(
    const torch::Tensor positions,    // [num_elements, N_POS_DIMS]
    const torch::Tensor grid,     // [total_hashmap_size, N_FEATURES_PER_LEVEL]
    const GridType grid_type,
    const uint32_t n_levels, 
    const uint32_t base_resolution, 
    const float per_level_scale, 
    const int log2_hashmap_size,
    const bool prepare_input_gradients
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grid_sample", &grid_sample, "");
    m.def("root_finding", &root_finding, "");
    py::enum_<InterpolationType>(m, "InterpolationType")
        .value("Nearest", InterpolationType::Nearest)
        .value("Linear", InterpolationType::Linear)
        .value("Smoothstep", InterpolationType::Smoothstep)
        .export_values();
    py::enum_<GridType>(m, "GridType")
        .value("Hash", GridType::Hash)
        .value("Dense", GridType::Dense)
        .value("Tiled", GridType::Tiled)
        .export_values();
}
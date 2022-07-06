#include "common.h"

inline std::string to_string(GridType grid_type) {
	switch (grid_type) {
		case GridType::Hash: return "Hash";
		case GridType::Dense: return "Dense";
		case GridType::Tiled: return "Tiled";
		default: throw std::runtime_error{std::string{"Invalid grid type"}};
	}
}

template <uint32_t N_DIMS>
__device__ uint32_t fast_hash(const uint32_t pos_grid[N_DIMS]) {
	static_assert(N_DIMS <= 7, "fast_hash can only hash up to 7 dimensions.");

	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
	constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i = 0; i < N_DIMS; ++i) {
		result ^= pos_grid[i] * primes[i];
	}

	return result;
}

template <uint32_t N_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__device__ uint32_t grid_index(
    const GridType grid_type, const uint32_t feature, const uint32_t hashmap_size, 
    const uint32_t grid_resolution, const uint32_t pos_grid[N_DIMS]
) {
	uint32_t stride = 1;
	uint32_t index = 0;

	// The second part of the loop condition is needed to avoid integer overflows in finer levels.
	#pragma unroll
	for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
		index += pos_grid[dim] * stride;
		stride *= grid_resolution;
	}

	if (grid_type == GridType::Hash && hashmap_size < stride) {
		index = fast_hash<N_DIMS>(pos_grid);
	}

	return (index % hashmap_size) * N_FEATURES_PER_LEVEL + feature;
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_grid_sample(
	const uint32_t num_elements,    // the number of points to be queried
	const uint32_t num_grid_features,   // n_level * N_FEATURES_PER_LEVEL
	const uint32_t* hashmap_offset_table,   // offset for each level
	const uint32_t base_resolution,
	const float log2_per_level_scale,
	const float quantize_threshold,    // default is 0.f;
	float max_level,    // default is 1000.f
	const float* __restrict__ max_level_gpu,    // default is nullptr
	const InterpolationType interpolation_type,
	const GridType grid_type,
	const T* __restrict__ grid,    // [total_params_all_level, N_FEATURES_PER_LEVEL]
	// MatrixView<const float> positions_in,
    const float* __restrict__ positions_in,  // [num_elements, 3]
	T* __restrict__ encoded_positions,
	float* __restrict__ dy_dx
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	// All zero return
	if (level >= max_level + 1e-3f) return;

	grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);

	float pos[N_POS_DIMS];
	float pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in[i * N_POS_DIMS + dim], &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative);
		}
	} else {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in[i * N_POS_DIMS + dim], &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, smoothstep, smoothstep_derivative);
		}
	}

	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
	};

	if (interpolation_type == InterpolationType::Nearest) {
		auto result = grid_val(pos_grid);

		if (encoded_positions) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
			}
		}

		// Gradient is zero when there's no interpolation.
		if (dy_dx) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0};
			}
		}

		return;
	}

	if (encoded_positions) {
		// N-linear interpolation
		vector_t<T, N_FEATURES_PER_LEVEL> result = {};

		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
			float weight = 1;
			uint32_t pos_grid_local[N_POS_DIMS];

			#pragma unroll
			for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
				if ((idx & (1<<dim)) == 0) {
					weight *= 1 - pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			auto val = grid_val(pos_grid_local);

			#pragma unroll
			for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
				float data = (float)((T*)&val)[feature];
				if (fabsf(data) < quantize_threshold) data = 0.f;
				((T*)&result)[feature] += (T)(weight * data);
			}
		}

		#pragma unroll
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
		}
	}

	// Gradient
	if (dy_dx) {
		vector_fullp_t<N_POS_DIMS> grads[N_FEATURES_PER_LEVEL] = {};

		#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			#pragma unroll
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
				float weight = scale;
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

					if ((idx & (1<<non_grad_dim)) == 0) {
						weight *= 1 - pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				auto val_left = grid_val(pos_grid_local);
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				auto val_right = grid_val(pos_grid_local);

				#pragma unroll
				for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
					grads[feature][grad_dim] += weight * ((float)val_right[feature] - (float)val_left[feature]) * pos_derivative[grad_dim];
				}
			}
		}

		#pragma unroll
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = grads[f];
		}
	}
}

// __global__ void kernel_softmax(
// 	const uint32_t n_features, 
// 	const uint32_t num_elements,    // the number of points to be queried
// 	const T* logits,	// [num_elements, dim]
// 	const float soft_blend,
// 	T* __restrict__ out
// ) {
// 	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (i >= num_elements) return;

// 	logits += i * n_features;
// 	out += i * n_features;

// 	T sum = 0;
// 	#pragma unroll
// 	for (uint32_t dim = 0; dim < n_features; ++dim) {
// 		sum += std::exp(logits[dim] * soft_blend);
// 	}

// 	#pragma unroll
// 	for (uint32_t dim = 0; dim < n_features; ++dim) {
// 		out[dim] = std::exp(logits[dim] * soft_blend) / sum;
// 	}
// 	return;
// }


inline __host__ __device__ float l2norm(float* vec) {
	float out = 0.f;
	#pragma unroll
	for (uint32_t i = 0; i < 3; ++i) {
		out += vec[i] * vec[i];
	}
	return sqrtf(out);
}

inline __host__ __device__ float vec_dot_vec(float* vec1, float* vec2) {
	float out = 0.f;
	#pragma unroll
	for (uint32_t i = 0; i < 3; ++i) {
		out += vec1[i] * vec2[i];
	}
	return out;
}

inline __host__ __device__ void mat_dot_vec(float* mat, float* vec, float* out) {
	#pragma unroll
	for (uint32_t i = 0; i < 3; ++i) {
		out[i] = 0.f;
		#pragma unroll
		for (uint32_t j = 0; j < 3; ++j) {
			out[i] += mat[i * 3 + j] * vec[j];
		}
	}
	return;
}

inline __host__ __device__ void subtract(float* vec1, float* vec2, float* out) {
	out[0] = vec1[0] - vec2[0];
	out[1] = vec1[1] - vec2[1];
	out[2] = vec1[2] - vec2[2];
	return;
}

__host__ __device__ void func_test(float* x, float* out) {
	// f(x) = x (x - 1)
	out[0] = x[0] * x[0] - x[0];
	out[1] = x[1] * x[1] - x[1];
	out[2] = x[2] * x[2] - x[2];
	return;
}

template <typename F>
__host__ __device__ bool broyden(
	F func,
	const float* jac,  // jacobian of func at x
	float* x  // init and out
) {
	// https://github.com/liruilong940607/implcarv/blob/main/models/root.py
	// https://github.com/kthohr/optim/blob/06c8596fa56a6e6ff6b6ec3dae8787b6ff77de3f/src/zeros/broyden.cpp

	uint32_t max_iters = 50;
	float tol = 1e-5;
	float dvg_thresh = 1.0;
	float eps = 1e-6;

	// printf ("x: [%f, %f, %f] \n", x[0], x[1], x[2]);

	float f[3];
	func(x, f);

	float err = l2norm(f);
	
	// printf ("f: [%f, %f, %f] \n", f[0], f[1], f[2]);
	// printf ("err: %f\n", err);
	
	if (err < tol) return true;  // converged
	if (err > dvg_thresh) return false;  // diverged

	// initial approx. to inverse Jacobian
	float J_inv[9] = {
		1.f, 0.f, 0.f, 
		0.f, 1.f, 0.f, 
		0.f, 0.f, 1.f
	};
	if (jac) {
		// computes the inverse of jac
		float det = (
			jac[0 * 3 + 0] * (jac[1 * 3 + 1] * jac[2 * 3 + 2] - jac[2 * 3 - 1] * jac[1 * 3 + 2]) -
			jac[0 * 3 + 1] * (jac[1 * 3 + 0] * jac[2 * 3 + 2] - jac[1 * 3 - 2] * jac[2 * 3 + 0]) +
			jac[0 * 3 + 2] * (jac[1 * 3 + 0] * jac[2 * 3 + 1] - jac[1 * 3 + 1] * jac[2 * 3 + 0])
		);
		if (fabsf(det) > 1e-6f) {
			float invdet = 1.f / det;
			J_inv[0 * 3 + 0] = (jac[1 * 3 + 1] * jac[2 * 3 + 2] - jac[2 * 3 + 1] * jac[1 * 3 + 2]) * invdet;
			J_inv[0 * 3 + 1] = (jac[0 * 3 + 2] * jac[2 * 3 + 1] - jac[0 * 3 + 1] * jac[2 * 3 + 2]) * invdet;
			J_inv[0 * 3 + 2] = (jac[0 * 3 + 1] * jac[1 * 3 + 2] - jac[0 * 3 + 2] * jac[1 * 3 + 1]) * invdet;
			J_inv[1 * 3 + 0] = (jac[1 * 3 + 2] * jac[2 * 3 + 0] - jac[1 * 3 + 0] * jac[2 * 3 + 2]) * invdet;
			J_inv[1 * 3 + 1] = (jac[0 * 3 + 0] * jac[2 * 3 + 2] - jac[0 * 3 + 2] * jac[2 * 3 + 0]) * invdet;
			J_inv[1 * 3 + 2] = (jac[1 * 3 + 0] * jac[0 * 3 + 2] - jac[0 * 3 + 0] * jac[1 * 3 + 2]) * invdet;
			J_inv[2 * 3 + 0] = (jac[1 * 3 + 0] * jac[2 * 3 + 1] - jac[2 * 3 + 0] * jac[1 * 3 + 1]) * invdet;
			J_inv[2 * 3 + 1] = (jac[2 * 3 + 0] * jac[0 * 3 + 1] - jac[0 * 3 + 0] * jac[2 * 3 + 1]) * invdet;
			J_inv[2 * 3 + 2] = (jac[0 * 3 + 0] * jac[1 * 3 + 1] - jac[1 * 3 + 0] * jac[0 * 3 + 1]) * invdet;
		}
	}
	
	float dx[3], x_new[3], df[3], f_new[3];
	float u[3], d[3], a[3], vT[3];
    for (uint32_t iter = 0; iter < max_iters; ++iter) {
		// update x
        mat_dot_vec(J_inv, f, dx);
        subtract(x, dx, x_new);
        x[0] = x_new[0];
		x[1] = x_new[1];
		x[2] = x_new[2];

		// update f
    	func(x, f_new);
        subtract(f_new, f, df);
        f[0] = f_new[0];
		f[1] = f_new[1];
		f[2] = f_new[2];

		err = l2norm(f);
		// printf ("[iter %d] err: %f\n", iter, err);
		if (err < tol) return true;  // converged
		if (err > dvg_thresh) return false;  // diverged

		// update J_inv for the next iteration.
        mat_dot_vec(J_inv, df, u);
        d[0] = -dx[0];
		d[1] = -dx[1];
		d[2] = -dx[2];
        subtract(d, u, a);
		float b = vec_dot_vec(d, u);
		b += ((b > 0) ? 1.f : -1.f) * eps;
		mat_dot_vec(J_inv, d, vT);
		
		J_inv[0] += vT[0] * a[0] / b;
		J_inv[1] += vT[1] * a[0] / b;
		J_inv[2] += vT[2] * a[0] / b;
		J_inv[3] += vT[0] * a[1] / b;
		J_inv[4] += vT[1] * a[1] / b;
		J_inv[5] += vT[2] * a[1] / b;
		J_inv[6] += vT[0] * a[2] / b;
		J_inv[7] += vT[1] * a[2] / b;
		J_inv[8] += vT[2] * a[2] / b;
	}        

	return false;
}

__global__ void kernel_root_finding(
	const uint32_t num_elements, 
	const float* x_jac,
	float* x_in_out,
	bool* success
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	// locate
	float* x = x_in_out + i * 3;
	const float* jac = x_jac ? x_jac + i * 3 * 3 : nullptr;

	success[i] = broyden(func_test, jac, x);

	return;
}

std::vector<torch::Tensor> root_finding(
    const torch::Tensor x_init, const torch::Tensor x_jac
) {
	const uint32_t num_elements = x_init.size(0);

	const uint32_t num_threads = 512;
    const uint32_t num_blocks = div_round_up(num_elements, num_threads);

	torch::Tensor x_root = x_init.clone(); 
	torch::Tensor success = torch::zeros(
		{num_elements}, x_init.options().dtype(torch::kBool)
	); 

	AT_DISPATCH_FLOATING_TYPES(
        x_init.scalar_type(),
        "root_finding",
        ([&]
         {kernel_root_finding<<<num_blocks, num_threads>>>(
			num_elements,
			x_jac.data_ptr<float>(), 
			x_root.data_ptr<float>(),
			success.data_ptr<bool>()
			);
         })
	);

	return {x_root, success};
}

std::vector<torch::Tensor> grid_sample(
    const torch::Tensor positions,    // [num_elements, N_POS_DIMS]
    const torch::Tensor grid,     // [total_hashmap_size, N_FEATURES_PER_LEVEL]
    const GridType grid_type,
    const uint32_t n_levels, 
    const uint32_t base_resolution, 
    const float per_level_scale, 
    const int log2_hashmap_size,
    const bool prepare_input_gradients
) {
    const uint32_t num_elements = positions.size(0);
    const uint32_t N_POS_DIMS = positions.size(1);
    const uint32_t total_hashmap_size = grid.size(0);
    const uint32_t N_FEATURES_PER_LEVEL = grid.size(1);
    const uint32_t n_features = n_levels * N_FEATURES_PER_LEVEL;
    
    // Idea: each block only takes care of _one_ hash level (but may iterate over multiple input elements).
    // This way, only one level of the hashmap needs to fit into caches at a time (and it reused for consecutive
    // elements) until it is time to process the next level.
    static constexpr uint32_t N_THREADS_HASHGRID = 512;
    const dim3 blocks_hashgrid = { div_round_up(num_elements, N_THREADS_HASHGRID), n_levels, 1 };

    uint32_t hashmap_offsets_table[n_levels + 1]; 
    uint32_t offset = 0;
    for (uint32_t i = 0; i < n_levels; ++i) {
        // Compute dense params required for the given level
        const float scale = exp2f(i * std::log2(per_level_scale)) * base_resolution - 1.0f;
        const uint32_t resolution = (uint32_t)(ceilf(scale)) + 1;

        uint32_t max_params = std::numeric_limits<uint32_t>::max()/2;
        uint32_t params_in_level = std::pow((float)resolution, N_POS_DIMS) > (float)max_params ? max_params : powi(resolution, N_POS_DIMS);

        // Make sure memory accesses will be aligned
        params_in_level = next_multiple(params_in_level, 8u);

        if (grid_type == GridType::Dense) {
            // No-op
        } else if (grid_type == GridType::Tiled) {
            // If tiled grid needs fewer params than dense, then use fewer and tile.
            params_in_level = std::min(params_in_level, powi(base_resolution, N_POS_DIMS));
        } else if (grid_type == GridType::Hash) {
            // If hash table needs fewer params than dense, then use fewer and rely on the hash.
            params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));
        } else {
            throw std::runtime_error{std::string{"GridEncoding: invalid grid type "} + to_string(grid_type)};
        }

        hashmap_offsets_table[i] = offset;
        offset += params_in_level;
    }
    hashmap_offsets_table[n_levels] = offset;

    torch::Tensor encoded_positions = torch::zeros({num_elements, n_features}, positions.options()); 
    torch::Tensor dy_dx;
    if (prepare_input_gradients) {
        dy_dx = torch::zeros({N_POS_DIMS * n_features, num_elements}, positions.options()); 
    }

    TORCH_CHECK(N_POS_DIMS == 3);
    TORCH_CHECK(N_FEATURES_PER_LEVEL == 2);

    AT_DISPATCH_FLOATING_TYPES(
        positions.scalar_type(),
        "grid_sample",
        ([&]
         {kernel_grid_sample<scalar_t, 3, 2><<<blocks_hashgrid, N_THREADS_HASHGRID>>>(
                num_elements,
                n_features,
                &hashmap_offsets_table[0],
                base_resolution,
                std::log2(per_level_scale),
                0.f,   // quantize_threshold
                1000.f,    // max_level
                nullptr,    // max_level_gpu
                InterpolationType::Linear,  // interpolation_type
                grid_type,
                grid.data_ptr<scalar_t>(),
                positions.data_ptr<float>(),
                encoded_positions.data_ptr<scalar_t>(),
                prepare_input_gradients ? dy_dx.data_ptr<float>() : nullptr
            );
        }));

    if (prepare_input_gradients) {
        return {encoded_positions, dy_dx};
    } else {
        return {encoded_positions};
    }
}
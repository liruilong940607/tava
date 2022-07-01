#include "common.cuh"

inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 8; }

inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
// inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
// inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;


inline __device__ float distance_to_next_voxel(
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    uint32_t res
) { // dda like step
	x, y, z = res * x, res * y, res * z;
	float tx = (floorf(x + 0.5f + 0.5f * sign(dir_x)) - x) * idir_x;
	float ty = (floorf(y + 0.5f + 0.5f * sign(dir_y)) - y) * idir_y;
	float tz = (floorf(z + 0.5f + 0.5f * sign(dir_z)) - z) * idir_z;
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(
    float t,
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    uint32_t res) {
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(
        x, y, z, dir_x, dir_y, dir_z, idir_x, idir_y, idir_z, res
    );
	do {
		t += STEPSIZE();
	} while (t < t_target);
	return t;
}

inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip, int grid_size) {
	return (grid_size * grid_size * grid_size) * mip;
}

__device__ uint32_t cascaded_grid_idx_at(
    float x, float y, float z, uint32_t mip, int grid_size
) {
	float mip_scale = scalbnf(1.0f, -mip);
    int ix = (int)((mip_scale * (x - 0.5f) + 0.5f) * grid_size);
    int iy = (int)((mip_scale * (y - 0.5f) + 0.5f) * grid_size);
    int iz = (int)((mip_scale * (z - 0.5f) + 0.5f) * grid_size);
    
	uint32_t idx = morton3D(
		clamp(ix, 0, grid_size-1),
		clamp(iy, 0, grid_size-1),
		clamp(iz, 0, grid_size-1)
	);
    // printf(
    //     "cascaded_grid_idx_at: (ix, iy, iz) = (%d, %d, %d); idx = %d; mip = %d\n", 
    //     ix, iy, iz, idx, mip
    // );
	return idx;
}

__device__ bool density_grid_occupied_at(
    float x, float y, float z, 
    const uint8_t* density_grid_bitfield,
    uint32_t mip, int grid_size
) {
	uint32_t idx = cascaded_grid_idx_at(x, y, z, mip, grid_size);
	return density_grid_bitfield[idx/8+grid_mip_offset(mip, grid_size)/8] & (1<<(idx%8));
}


inline __device__ int mip_from_pos(float x, float y, float z) {
	int exponent;
    float maxval = fmaxf(fmaxf(fabsf(x - 0.5f), fabsf(y - 0.5f)), fabsf(z - 0.5f));
	frexpf(maxval, &exponent);
	return min(NERF_CASCADES()-1, max(0, exponent+1));
}


inline __device__ int mip_from_dt(
    float x, float y, float z, float dt, int grid_size
) {
	int mip = mip_from_pos(x, y, z);
	dt *= 2 * grid_size;
	if (dt<1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(NERF_CASCADES()-1, max(exponent, mip));
}

template <typename scalar_t>
inline __host__ __device__ bool aabb_contains(
    const scalar_t *aabb, float x, float y, float z
) {
    return
        x >= aabb[0] && x <= aabb[3] &&
        y >= aabb[1] && y <= aabb[4] &&
        z >= aabb[2] && z <= aabb[5];
}


template <typename scalar_t>
__host__ __device__ void ray_aabb_intersect(
    const scalar_t* rays_o,
    const scalar_t* rays_d,
    const scalar_t* aabb,
    scalar_t* nears,
    scalar_t* fars
) {
    // aabb is [xmin, ymin, zmin, xmax, ymax, zmax]
    float tmin = (aabb[0] - rays_o[0]) / rays_d[0];
    float tmax = (aabb[3] - rays_o[0]) / rays_d[0];
    if (tmin > tmax) swapf(tmin, tmax);

    float tymin = (aabb[1] - rays_o[1]) / rays_d[1];
    float tymax = (aabb[4] - rays_o[1]) / rays_d[1];
    if (tymin > tymax) swapf(tymin, tymax);

    if (tmin > tymax || tymin > tmax){
        nears[0] = fars[0] = std::numeric_limits<float>::max();
        return;
    }

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    float tzmin = (aabb[2] - rays_o[2]) / rays_d[2];
    float tzmax = (aabb[5] - rays_o[2]) / rays_d[2];
    if (tzmin > tzmax) swapf(tzmin, tzmax);

    if (tmin > tzmax || tzmin > tmax){
        nears[0] = fars[0] = std::numeric_limits<float>::max();
        return;
    }

    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    nears[0] = tmin;
    fars[0] = tmax;
    return;
}

template <typename scalar_t>
__global__ void kernel_generate_training_samples(
    const uint32_t n_rays,
    const scalar_t* rays_o,
    const scalar_t* rays_d,
    const scalar_t* aabb,
    scalar_t* nears,
    scalar_t* fars,
    const uint32_t grid_size,  // default is 128
    const uint8_t* density_bitfield,
    int* numsteps_counter,  // total samples.
    const int max_samples,
    int* rays_counter,  // total rays.
    int* ray_indices_out,  // output ray indices.
    scalar_t* positions_out,  // output samples
    scalar_t* dirs_out,  // output dirs
    scalar_t* deltas_out  // output delta t
) {
    // TODO(ruilongli): check those + 0.5f operation
    // what scale should the input be?

    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    rays_o += i * 3;
    rays_d += i * 3;
    nears += i;
    fars += i;

    // ray aabb test: The near distance prevents learning of camera-specific fudge 
    // right in front of the camera.
    ray_aabb_intersect(rays_o, rays_d, aabb, nears, fars);
    nears[0] = fmaxf(nears[0], 0.0f);

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float startt = nears[0], endt = fars[0];
    // if (i == 0) {
	//     printf("thread %d: startt %f; endt %f;\n", i, startt, endt);
    // }

    // first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;

	while (t < endt && j < NERF_STEPS()) {
        // current point
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;
        
        // if (i == 0) {
        //     printf("thread %d: t %f, contains %d;\n", i, t, aabb_contains(aabb, x, y, z));
        // }
        if (!aabb_contains(aabb, x, y, z)) break;
        
        float dt = STEPSIZE();
		uint32_t mip = mip_from_dt(x, y, z, dt, grid_size);
		// if (i == 0) {
        //     printf(
        //         "thread %d: mip %d, density_grid_occupied_at %f, advance_to_next_voxel %f;\n", 
        //         i, 
        //         mip, 
        //         density_grid_occupied_at(x, y, z, density_bitfield, mip, grid_size),
        //         advance_to_next_voxel(
        //             t, x, y, z, dx, dy, dz, rdx, rdy, rdz, 128
        //         )
        //     );
        // }
        if (density_grid_occupied_at(x, y, z, density_bitfield, mip, grid_size)) {
            ++j;
			t += dt;
		}
        else {
			uint32_t res = grid_size >> mip;
			t = advance_to_next_voxel(
                t, x, y, z, dx, dy, dz, rdx, rdy, rdz, res
            );
            // if (i == 0) {
            //     printf("res %d; t %f \n", res, t);
            // }
		}
        // if (i == 0) {
	    //     printf("thread %d: t: %f, j: %d\n", i, t, j);
        // }
	}
    if (j == 0) return;
    // if (i == 0) {
	//     printf("thread %d: j: %d\n", i, j);
    // }

    uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);
	// printf("thread %d: base %d, numsteps %d\n", i, base, numsteps);
    if (base + numsteps > max_samples) return;
	
    // locate
    positions_out += base * 3;
    dirs_out += base * 3;
    deltas_out += base;

    uint32_t ray_idx = atomicAdd(rays_counter, 1);

	ray_indices_out[ray_idx] = i;
	// rays_out_unnormalized[ray_idx] = ray_unnormalized;
	// numsteps_out[ray_idx*2+0] = numsteps;
	// numsteps_out[ray_idx*2+1] = base;

	t = startt;
	j = 0;
    while (t < endt && j < NERF_STEPS()) {
        // current point
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;
        if (!aabb_contains(aabb, x, y, z)) break;
        float dt = STEPSIZE();
		uint32_t mip = mip_from_dt(x, y, z, dt, grid_size);
        if (density_grid_occupied_at(x, y, z, density_bitfield, mip, grid_size)) {
            positions_out[j * 3 + 0] = x;
            positions_out[j * 3 + 1] = y;
            positions_out[j * 3 + 2] = z;
            dirs_out[j * 3 + 0] = dx,
            dirs_out[j * 3 + 1] = dy,
            dirs_out[j * 3 + 2] = dz,
            deltas_out[j * 3 + 0] = dt;            
            // coords_out(j)->set_with_optional_extra_dims(
            //     warp_position(x, y, z, aabb), 
            //     warp_direction(dx, dy, dz), 
            //     warp_dt(dt), 
            //     extra_dims, 
            //     coords_out.stride_in_bytes
            // );
            ++j;
			t += dt;
		}
        else {
			uint32_t res = grid_size >> mip;
			t = advance_to_next_voxel(
                t, x, y, z, dx, dy, dz, rdx, rdy, rdz, res
            );
		}
	}

    return;
}

std::vector<torch::Tensor> generate_training_samples(
    torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor aabb,
    torch::Tensor density_bitfield, int max_samples
) {
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(aabb);
    TORCH_CHECK(rays_o.ndimension() == 2);
    TORCH_CHECK(rays_d.ndimension() == 2);
    TORCH_CHECK(aabb.ndimension() == 1);
    // TORCH_CHECK(density_bitfield.ndimension() == 4);
    const uint32_t n_rays = rays_o.size(0);
    // const uint32_t n_mip = density_bitfield.size(0);
    const uint32_t grid_size = 128;  //density_bitfield.size(1);

    const int cuda_n_threads = std::min<int>(n_rays, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, cuda_n_threads);

    torch::Tensor nears = torch::empty({n_rays}, rays_o.options());
    torch::Tensor fars = torch::empty({n_rays}, rays_o.options());

    torch::Tensor numsteps_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor rays_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor ray_indices = torch::empty(
        {n_rays}, rays_o.options().dtype(torch::kInt32));

    torch::Tensor positions = torch::empty({max_samples, 3}, rays_o.options());
    torch::Tensor dirs = torch::empty({max_samples, 3}, rays_o.options());
    torch::Tensor deltas = torch::empty({max_samples}, rays_o.options());

    AT_DISPATCH_FLOATING_TYPES(
        rays_o.scalar_type(),
        "generate_training_samples",
        ([&]
         { kernel_generate_training_samples<<<blocks, cuda_n_threads>>>(
                n_rays,
                rays_o.data_ptr<scalar_t>(),
                rays_d.data_ptr<scalar_t>(),
                aabb.data_ptr<scalar_t>(),
                nears.data_ptr<scalar_t>(),  // output
                fars.data_ptr<scalar_t>(),  // output
                grid_size,
                density_bitfield.data_ptr<uint8_t>(),
                numsteps_counter.data_ptr<int>(),  // total samples.
                max_samples,
                rays_counter.data_ptr<int>(),  // total rays.
                ray_indices.data_ptr<int>(),  // output ray indices.
                positions.data_ptr<scalar_t>(),  // output samples
                dirs.data_ptr<scalar_t>(),  // output dirs
                deltas.data_ptr<scalar_t>()  // output delta t
            ); 
        }));

    return {positions, dirs, deltas, nears, fars};
}
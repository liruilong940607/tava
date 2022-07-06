/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   common.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// #include <cmath>
// #include <array>
// #include <iostream>
// #include <sstream>
// #include <stdexcept>
// #include <string>
// #include <cassert>
// #include <cstdint>
#include <cstdio>

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT2 = 1.41421356237309504880f;

enum class InterpolationType {
	Nearest,
	Linear,
	Smoothstep,
};

enum class GridType {
	Hash,
	Dense,
	Tiled,
};


template <typename T, uint32_t N_ELEMS>
struct vector_t {
	__host__ __device__ T& operator[](uint32_t idx) {
		return data[idx];
	}

	__host__ __device__ T operator [](uint32_t idx) const {
		return data[idx];
	}

	T data[N_ELEMS];
	static constexpr uint32_t N = N_ELEMS;
};

template <uint32_t N_FLOATS>
using vector_fullp_t = vector_t<float, N_FLOATS>;

template <uint32_t N_HALFS>
using vector_halfp_t = vector_t<__half, N_HALFS>;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

__host__ __device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

__host__ __device__ inline uint64_t expand_bits(uint64_t w)  {
	w &=                0x00000000001fffff;
	w = (w | w << 32) & 0x001f00000000ffff;
	w = (w | w << 16) & 0x001f0000ff0000ff;
	w = (w | w <<  8) & 0x010f00f00f00f00f;
	w = (w | w <<  4) & 0x10c30c30c30c30c3;
	w = (w | w <<  2) & 0x1249249249249249;
	return w;
}

__host__ __device__ inline uint64_t morton3D_64bit(uint32_t x, uint32_t y, uint32_t z)  {
	return ((expand_bits((uint64_t)x)) | (expand_bits((uint64_t)y) << 1) | (expand_bits((uint64_t)z) << 2));
}

__device__ inline float smoothstep(float val) {
	return val*val*(3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6*val*(1.0f - val);
}

__device__ inline float smoothstep_2nd_derivative(float val) {
	return 6.0f - 12.0f * val;
}

__device__ inline float identity_fun(float val) {
	return val;
}

__device__ inline float identity_derivative(float val) {
	return 1.0f;
}

__device__ inline float identity_2nd_derivative(float val) {
	return 0.0f;
}

template <typename F, typename FPRIME, typename FPRIMEPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, float* pos_2nd_derivative, uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative, FPRIMEPRIME interpolation_fun_2nd_derivative) {
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos_2nd_derivative = interpolation_fun_2nd_derivative(*pos);
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F, typename FPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative) {
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F>
__device__ inline void pos_fract(const float input, float* pos, uint32_t* pos_grid, float scale, F interpolation_fun) {
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos = interpolation_fun(*pos);
}

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
__host__ __device__ T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

inline uint32_t powi(uint32_t base, uint32_t exponent) {
	uint32_t result = 1;
	for (uint32_t i = 0; i < exponent; ++i) {
		result *= base;
	}

	return result;
}
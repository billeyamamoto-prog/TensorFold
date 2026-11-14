// Device code of nvfp4/qmmf.cu (lines 14-298, comments and ATen dropped), its FP4, FP8 and FP8G modes plus BF16.

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/qmm_frag.cuh"

namespace tf_qmmf {

using namespace qmm_frag;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4 };

constexpr int GS = 64;

__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
    return r;
}

// UE8M0 has no zero encoding: byte 0 is 2^-127, byte 255 is NaN.
__device__ __forceinline__ float e8m0f(uint8_t b) {
    return __uint_as_float(b == 0 ? 0x00400000u : b == 255 ? 0x7FC00000u : uint32_t(b) << 23);
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

using tf_common::e4m3f;

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int BARRIERS = MODE == BF16 && BN > 64 ? 128 : 0;
    static constexpr int SMEM = (STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS) + BARRIERS;
};

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__device__ __forceinline__ void qmmf_body(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group, int ldy) {
#define TF_QMMF_STRIDED
#include "qmmf_body.cuh"
#undef TF_QMMF_STRIDED
}


template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__global__ void __launch_bounds__(WM * WN * 32) qmmf_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group) {
#include "qmmf_body.cuh"
}

// The shared down has one K slice; only the output address changes from the existing BF16-weight product.
#define TF_SHARED_DOWN(NAME, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
        const __nv_bfloat16* x, const unsigned char* w, float scale, float* out, \
        int M, int N, int K, int npad, int ldx, int group, int ldy) { \
    qmmf_body<BF16, BM, BN, 1, 4, 4, true, false, false>( \
        x, w, nullptr, scale, out, nullptr, M, N, K, 1, npad, ldx, group, ldy); \
}
TF_SHARED_DOWN(tf_qmmf_shared16_half, 16, 32)
TF_SHARED_DOWN(tf_qmmf_shared16, 16, 64)
TF_SHARED_DOWN(tf_qmmf_shared32, 32, 64)
TF_SHARED_DOWN(tf_qmmf_shared64, 64, 64)
#undef TF_SHARED_DOWN

template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK,
                              float scale) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    acc = acc * scale;
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);

// The instances the Zig launchers load by name (the Python host instantiates these through its launch switch).
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, true, false, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, true, true, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, true, false, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, true, true, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, true, false, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, true, true, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, true, false, true>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);







template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);

// Half tiles (32 columns a block) of the 16-row tile: the 64-column tile's sums, twice the blocks for a small grid.
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);

template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
// Prompt tiles retain the same m16n8k16 chains and fused slice boundaries as the 64x64 instances.
#define BF16_PROMPT(F32, FUSE) \
    template __global__ void qmmf_kernel<BF16, 128, 128, 2, 4, 3, F32, false, FUSE>( \
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
BF16_PROMPT(false, false)
BF16_PROMPT(false, true)
BF16_PROMPT(true, false)
BF16_PROMPT(true, true)
#undef BF16_PROMPT

// Unsplit K32 needs no inter-slice partial buffer or bulk-copy barrier storage.
struct PromptK32Tile {
    static constexpr int THREADS = 256, MT = 4, NT = 4;
    static constexpr int ROW = 64, CHUNKS = ROW / 16;
    static constexpr int X = 128 * ROW, W = 128 * 32 * 2, S = 0;
    static constexpr int STAGE = (X + W + 127) / 128 * 128;
    static constexpr int BARRIERS = 0, SMEM = 3 * STAGE;
};

// Unsplit K32 staging keeps the parent's packed weights, K16 MMA chain and final output rounding.
static_assert(PromptK32Tile::SMEM == 49152, "K32 128x128 shared bytes");

template <bool F32>
__global__ void __launch_bounds__(256, 2) qmmf_prompt_k32_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group) {
    constexpr int MODE = BF16, BM = 128, BN = 128, WM = 2, WN = 4, STAGES = 3, BK = 32;
    constexpr bool CLUSTER = false, FUSE = false;
#define TF_QMMF_K32
#include "qmmf_body.cuh"
#undef TF_QMMF_K32
}
#define BF16_K32(F32) \
    template __global__ void qmmf_prompt_k32_kernel<F32>( \
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
BF16_K32(false)
BF16_K32(true)
#undef BF16_K32

template __global__ void reduce_kernel<true>(const float* __restrict__, void* __restrict__, long long, int, float);
template __global__ void reduce_kernel<false>(const float* __restrict__, void* __restrict__, long long, int, float);


// ---- MXFP8 prompt rows on a 128 x 128 tile (relaxed): the block scale rides in the B fragment, one fp32 chain over K ----
//
// The exact kernels above add each 32-input group's mma sum to the accumulator with one fmaf per element. Here the
// e4m3 code times its UE8M0 scale is formed exactly in bf16 (a power-of-two multiply), so a warp issues only mma
// instructions over the whole of K. Rows are summed in the same order whatever the chunk holds (a row's bits do not
// depend on M or on the tile it falls in), but they are not the exact kernels' bits; callers gate it (TF_FLASHNEXT_PROMPT_MX128).
__device__ __forceinline__ uint32_t fp8pair_unit(uint32_t w, uint32_t unit) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(unit), "r"(0x80008000u));
    return r;
}

template <int BM, int BN, int WM, int WN, int STAGES, bool F32>
__device__ __forceinline__ void prompt_mx_body(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {
    using T = Tile<MXFP8, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            if (n0 + t * 64 < npad) cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        constexpr int PER = T::S / (BN / 64);
        for (int c = tid; c < T::S / 16; c += T::THREADS) {
            const int t = c / (PER / 16), off = c % (PER / 16);
            if (n0 + t * 64 < npad) cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + t) * KG + g) * PER + off * 16);
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int g = 0; g < KG; ++g) {
        wait<STAGES - 2>();
        __syncthreads();
        if (g + STAGES - 1 < KG) load((g + STAGES - 1) % STAGES, g + STAGES - 1);
        commit();
        const unsigned char* p = stage(g % STAGES);
        const unsigned char* pw = p + T::X;
        const uint8_t* ps = p + T::X + T::W;
        uint4 wq[T::NT];
        uint32_t sv[T::NT][2];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            const int col = (jj & 7) * 8 + (lane >> 2);
            wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
            const uint32_t e2 = *reinterpret_cast<const uint16_t*>(ps + (jj >> 3) * 128 + col * 2);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t u = (((e2 >> (8 * h)) & 0xFFu) + 120u) << 7;  // 2^(e - 7), the 2^-120 of the shifted code folded in
                sv[j][h] = u | (u << 16);
            }
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t word = comp(wq[j], kt);
                const uint32_t unit = sv[j][kt >> 1];
                const uint32_t b0 = fp8pair_unit(word & 0xFFFFu, unit);
                const uint32_t b1 = fp8pair_unit(word >> 16, unit);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

#if defined(TF_NATIVE_FP_DECODE)
// tf-cuda-test native-fp: e2m1_pairs == fp4pair on all words, codes x e4m3 scales; out: misses, first, checks
extern "C" __global__ void tf_qmmf_native_fp4_check(unsigned long long* __restrict__ out) {
#if TF_FP_NATIVE
    const unsigned long long n = 1ull << 33, step = 1ull * gridDim.x * blockDim.x;
    unsigned long long bad = 0, first = ~0ull, seen = 0;
    for (unsigned long long i = 1ull * blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        const uint32_t w = static_cast<uint32_t>(i >> 1);
        const int q = static_cast<int>(i & 1);
        uint32_t b0, b1;
        tf_common::e2m1_pairs(w, q, b0, b1);
        ++seen;
        if (b0 != tf_common::fp4pair(w, 8 * q) || b1 != tf_common::fp4pair(w, 8 * q + 4)) {
            ++bad;
            first = i < first ? i : first;
        }
    }
    const unsigned long long t = 1ull * blockIdx.x * blockDim.x + threadIdx.x;
    if (t < 16 * 256) {
        const uint32_t c = static_cast<uint32_t>(t >> 8), s = static_cast<uint32_t>(t & 255);
        const uint32_t hw = tf_common::e2m1x2(c | (c << 4)), sw = tf_common::fp4pair(c | (c << 16), 0);
        const float sf = tf_common::e4m3f(s);
        const uint32_t sp = tf_common::fp8pair_unit(s | (s << 8), 0x7B807B80u);
        const float h = __uint_as_float(hw << 16) * sf, g = __uint_as_float(sw << 16) * sf;
        ++seen;
        // fp32 products compare as bits (NaN scale included); so do the bf16x2 products by the e4m3 scale pair
        if (__float_as_uint(h) != __float_as_uint(g) || tf_common::mul2(hw, sp) != tf_common::mul2(sw, sp)) {
            ++bad;
            first = ((1ull << 40) + t) < first ? (1ull << 40) + t : first;
        }
    }
    if (bad) {
        atomicAdd(&out[0], bad);
        atomicMin(&out[1], first);
    }
    atomicAdd(&out[2], seen);
#else
    if (blockIdx.x == 0 && threadIdx.x < 3) out[threadIdx.x] = ~0ull;
#endif
}
#endif
#define TF_PROMPT_MX(NAME, BM, BN, WM, WN, ST, F32)                                                                    \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(                                                   \
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,     \
        float scale, void* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {                      \
        tf_qmmf::prompt_mx_body<BM, BN, WM, WN, ST, F32>(x, w, bs, scale, out, M, N, K, npad, ldx, group);                   \
    }

}  // namespace tf_qmmf

// Names the Zig launcher loads (qmmf.zig Lane.prompt_mx): block rows x columns, warps, stages, bf16 or fp32 out.
TF_PROMPT_MX(tf_qmmf_prompt_mx_bf16, 128, 128, 2, 4, 3, false)
TF_PROMPT_MX(tf_qmmf_prompt_mx_f32, 128, 128, 2, 4, 3, true)

// GLP refusal-direction projection on Flash Next's hyper-connection residual streams.
// One block per row, 256 threads: a row is the materialized post-layer stream, S streams of D
// columns flat (HC-outer / H-inner; S*D is the derivation width the vector was fit at). The
// layer's unit direction is that same width and the row's single component along it leaves:
//   h ← h − α·(h · d̂)·d̂   over the whole row, h bf16, the dot in fp32.
// The block reduces the dot in shared memory, then applies the update. The kernel reads H in
// place; PSS, the sums and every other row's bytes are untouched.

#include <cuda_bf16.h>
#include <stdint.h>

#include "glue_math.cuh"

namespace fn_glp {

__device__ __forceinline__ float rbf(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

} // namespace fn_glp

using namespace fn_glp;

// glue._glp_project, program (rows, 1 block): one unit direction S*D wide, one dot across the row.
extern "C" __global__ void __launch_bounds__(256) fn_glp_project(
        const float* __restrict__ DIR, __nv_bfloat16* __restrict__ H,
        float alpha, int D, int S) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int n = D * S;
    __nv_bfloat16* h = H + static_cast<long long>(r) * n;
    __shared__ float red[8];
    __shared__ float dot_sh;

    // The row's dot with the unit direction: fp32 products, a fixed-order block sum.
    float acc = 0.0f;
    for (int d = t; d < n; d += 256) {
        acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(h[d]), DIR[d]));
    }
    const int warp = t >> 5;
    const int lane = t & 31;
    for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (t == 0) {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) v = __fadd_rn(v, red[w]);
        dot_sh = __fmul_rn(alpha, v);
    }
    __syncthreads();
    const float k = dot_sh;
    if (k != 0.0f) {
        for (int d = t; d < n; d += 256) {
            const float nv = __fsub_rn(__bfloat162float(h[d]), __fmul_rn(k, DIR[d]));
            h[d] = __float2bfloat16_rn(rbf(nv));
        }
    }
}

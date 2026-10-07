#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpsPerBlock = 2;
constexpr int kThreads = 32 * kWarpsPerBlock;
constexpr int kTileN = 8;
constexpr int kTilesPerWarp = 4;
constexpr int kRowBlock = kTileN * kTilesPerWarp;
constexpr int kRowsPerBlock = kWarpsPerBlock * kRowBlock;
constexpr int kWordsPerGroupBlock = kRowBlock * 8;
constexpr int kMaxRows = 64;
constexpr int kTargetWarps = 2048;
constexpr int kMaxSplitK = 16;

__device__ __forceinline__ uint32_t bf16_bits(int value) {
    return value < 0 ? 0xBF80u : (value > 0 ? 0x3F80u : 0u);
}

// Turing (SM75) has FP16 Tensor Cores but no BF16 Tensor Cores.  The original
// kernel feeds BF16 fragments directly to the Ampere+ BF16 MMA instruction.
// For SM75 we convert each BF16 pair to FP16 in registers and use the
// corresponding FP16 Tensor Core instruction.  Accumulation remains FP32.
__device__ __forceinline__ uint16_t bf16_to_fp16_bits(uint16_t bits) {
    const float value = __uint_as_float(static_cast<uint32_t>(bits) << 16);
    return __half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ uint32_t bf16_pair_to_fp16_pair(uint32_t bits) {
    const uint16_t lo = bf16_to_fp16_bits(static_cast<uint16_t>(bits));
    const uint16_t hi = bf16_to_fp16_bits(static_cast<uint16_t>(bits >> 16));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint32_t fp16_bits(int value) {
    return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(static_cast<float>(value))));
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#endif

__device__ __forceinline__ void mma_fp16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int kMBlocks, bool kAtomic>
__global__ void __launch_bounds__(kThreads) ternary_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ w,
    const __half* __restrict__ s,
    void* __restrict__ y,
    int rows,
    int N,
    int K,
    int groups_per_split) {
    __shared__ uint32_t lut[16];
    if (threadIdx.x < 16) {
#if __CUDA_ARCH__ >= 800
        lut[threadIdx.x] = bf16_bits((threadIdx.x & 3) - 1) | (bf16_bits((threadIdx.x >> 2) - 1) << 16);
#else
        lut[threadIdx.x] = fp16_bits((threadIdx.x & 3) - 1) | (fp16_bits((threadIdx.x >> 2) - 1) << 16);
#endif
    }
    __syncthreads();
    const int groups = K / 128;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int n0 = (blockIdx.x * kWarpsPerBlock + warp) * kRowBlock;
    if (n0 >= N) return;
    const int g_begin = blockIdx.y * groups_per_split;
    const int g_end = min(groups, g_begin + groups_per_split);
    const uint32_t* wblock = w + static_cast<size_t>(n0 / kRowBlock) * groups * kWordsPerGroupBlock + gid * 8;
    const __half* sblock = s + static_cast<size_t>(n0 / kRowBlock) * groups * kRowBlock + 2 * tig;
    float acc[kTilesPerWarp][kMBlocks][4];
#pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t)
#pragma unroll
        for (int b = 0; b < kMBlocks; ++b)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[t][b][i] = 0.f;
    const __nv_bfloat16* xlo[kMBlocks];
    bool a_lo[kMBlocks];
    bool a_hi[kMBlocks];
#pragma unroll
    for (int b = 0; b < kMBlocks; ++b) {
        const int m = b * 16 + gid;
        a_lo[b] = m < rows;
        a_hi[b] = m + 8 < rows;
        xlo[b] = x + static_cast<size_t>(min(m, rows - 1)) * K + 2 * tig;
    }
    uint4 pk[kTilesPerWarp][2];
    auto load_group = [&](int g, uint4 (*dst)[2]) {
        const uint32_t* src = wblock + static_cast<size_t>(g) * kWordsPerGroupBlock;
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t) {
#pragma unroll
            for (int h = 0; h < 2; ++h) dst[t][h] = *reinterpret_cast<const uint4*>(src + t * kTileN * 8 + h * 4);
        }
    };
    if (g_begin < g_end) load_group(g_begin, pk);
    for (int g = g_begin; g < g_end; ++g) {
        uint4 cur[kTilesPerWarp][2];
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t) {
            cur[t][0] = pk[t][0];
            cur[t][1] = pk[t][1];
        }
        if (g + 1 < g_end) load_group(g + 1, pk);
        float ctmp[kTilesPerWarp][kMBlocks][4];
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t)
#pragma unroll
            for (int b = 0; b < kMBlocks; ++b)
#pragma unroll
                for (int i = 0; i < 4; ++i) ctmp[t][b][i] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = g * 128 + h * 64 + j * 16;
                uint32_t a[kMBlocks][4];
#pragma unroll
                for (int b = 0; b < kMBlocks; ++b) {
                    const __nv_bfloat16* xp = xlo[b] + k;
                    a[b][0] = a_lo[b] ? *reinterpret_cast<const uint32_t*>(xp) : 0u;
                    a[b][2] = a_lo[b] ? *reinterpret_cast<const uint32_t*>(xp + 8) : 0u;
                    a[b][1] = a_hi[b] ? *reinterpret_cast<const uint32_t*>(xp + static_cast<size_t>(8) * K) : 0u;
                    a[b][3] = a_hi[b] ? *reinterpret_cast<const uint32_t*>(xp + static_cast<size_t>(8) * K + 8) : 0u;
                }
#pragma unroll
                for (int t = 0; t < kTilesPerWarp; ++t) {
                    const uint32_t word = j == 0 ? cur[t][h].x : (j == 1 ? cur[t][h].y : (j == 2 ? cur[t][h].z : cur[t][h].w));
                    uint32_t b0 = lut[(word >> (4 * tig)) & 0xFu];
                    uint32_t b1 = lut[(word >> (4 * tig + 16)) & 0xFu];
#pragma unroll
                    for (int b = 0; b < kMBlocks; ++b) {
#if __CUDA_ARCH__ >= 800
                        mma_bf16(ctmp[t][b], a[b][0], a[b][1], a[b][2], a[b][3], b0, b1);
#else
                        mma_fp16(ctmp[t][b],
                                 bf16_pair_to_fp16_pair(a[b][0]),
                                 bf16_pair_to_fp16_pair(a[b][1]),
                                 bf16_pair_to_fp16_pair(a[b][2]),
                                 bf16_pair_to_fp16_pair(a[b][3]),
                                 b0, b1);
#endif
                    }
                }
            }
        }
        const __half* sg = sblock + static_cast<size_t>(g) * kRowBlock;
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t) {
            const __half2 sc = *reinterpret_cast<const __half2*>(sg + t * kTileN);
            const float sc0 = __low2float(sc);
            const float sc1 = __high2float(sc);
#pragma unroll
            for (int b = 0; b < kMBlocks; ++b) {
                acc[t][b][0] = fmaf(sc0, ctmp[t][b][0], acc[t][b][0]);
                acc[t][b][1] = fmaf(sc1, ctmp[t][b][1], acc[t][b][1]);
                acc[t][b][2] = fmaf(sc0, ctmp[t][b][2], acc[t][b][2]);
                acc[t][b][3] = fmaf(sc1, ctmp[t][b][3], acc[t][b][3]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t) {
        const int nc0 = n0 + t * kTileN + 2 * tig;
        const int nc1 = nc0 + 1;
#pragma unroll
        for (int b = 0; b < kMBlocks; ++b) {
            const size_t m_lo = static_cast<size_t>(b * 16 + gid);
            const size_t m_hi = m_lo + 8;
            if constexpr (kAtomic) {
                float* out = static_cast<float*>(y) + static_cast<size_t>(blockIdx.y) * rows * N;
                if (a_lo[b]) out[m_lo * N + nc0] = acc[t][b][0];
                if (a_lo[b]) out[m_lo * N + nc1] = acc[t][b][1];
                if (a_hi[b]) out[m_hi * N + nc0] = acc[t][b][2];
                if (a_hi[b]) out[m_hi * N + nc1] = acc[t][b][3];
            } else {
                __nv_bfloat16* out = static_cast<__nv_bfloat16*>(y);
                if (a_lo[b]) *reinterpret_cast<__nv_bfloat162*>(out + m_lo * N + nc0) = __floats2bfloat162_rn(acc[t][b][0], acc[t][b][1]);
                if (a_hi[b]) *reinterpret_cast<__nv_bfloat162*>(out + m_hi * N + nc0) = __floats2bfloat162_rn(acc[t][b][2], acc[t][b][3]);
            }
        }
    }
}

__global__ void reduce_split_kernel(
    const float* __restrict__ partial,
    __nv_bfloat16* __restrict__ y,
    int elements,
    int split) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    float sum = 0.f;
    for (int p = 0; p < split; ++p) sum += partial[static_cast<size_t>(p) * elements + i];
    y[i] = __float2bfloat16(sum);
}

template <int kMBlocks>
void launch_mma(const __nv_bfloat16* x, const uint32_t* w, const __half* s, void* y, int rows, int N, int K, int split, int groups_per_split, cudaStream_t stream) {
    const dim3 grid(N / kRowsPerBlock, split);
    if (split == 1) ternary_mma_kernel<kMBlocks, false><<<grid, kThreads, 0, stream>>>(x, w, s, y, rows, N, K, groups_per_split);
    else ternary_mma_kernel<kMBlocks, true><<<grid, kThreads, 0, stream>>>(x, w, s, y, rows, N, K, groups_per_split);
}

constexpr int kHadamardBlock = 1024;
constexpr int kHadamardThreads = 256;

__global__ void __launch_bounds__(kHadamardThreads) signed_hadamard_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ signs,
    __nv_bfloat16* __restrict__ y,
    int K) {
    __shared__ float buf[kHadamardBlock];
    const size_t base = static_cast<size_t>(blockIdx.y) * K + static_cast<size_t>(blockIdx.x) * kHadamardBlock;
    const int col0 = blockIdx.x * kHadamardBlock;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * kHadamardThreads;
        buf[idx] = __bfloat162float(x[base + idx]) * signs[col0 + idx];
    }
    __syncthreads();
    float values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) values[i] = buf[threadIdx.x + i * kHadamardThreads];
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float other = __shfl_xor_sync(0xffffffffu, values[i], stride);
            values[i] = (threadIdx.x & stride) ? other - values[i] : values[i] + other;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) buf[threadIdx.x + i * kHadamardThreads] = values[i];
    __syncthreads();
    for (int stride = 32; stride < kHadamardBlock; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = threadIdx.x + i * kHadamardThreads;
            const int a = ((p / stride) * stride * 2) + (p % stride);
            const int b = a + stride;
            const float u = buf[a];
            const float v = buf[b];
            buf[a] = u + v;
            buf[b] = u - v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * kHadamardThreads;
        y[base + idx] = __float2bfloat16(buf[idx] * 0.03125f);
    }
}

}  // namespace

torch::Tensor signed_hadamard(torch::Tensor x, torch::Tensor signs) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous(), "x must be contiguous bf16 on cuda");
    TORCH_CHECK(signs.dtype() == torch::kFloat32 && signs.is_contiguous(), "signs must be contiguous fp32");
    const int rows = x.size(0);
    const int K = x.size(1);
    TORCH_CHECK(K % kHadamardBlock == 0 && signs.numel() == K, "K must be a multiple of 1024 and match signs");
    auto y = torch::empty_like(x);
    const dim3 grid(K / kHadamardBlock, rows);
    signed_hadamard_kernel<<<grid, kHadamardThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        signs.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        K);
    return y;
}

torch::Tensor ternary_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous(), "x must be contiguous bf16 on cuda");
    TORCH_CHECK(w.dtype() == torch::kInt32 && w.is_contiguous(), "weight must be contiguous int32 in blocked layout");
    TORCH_CHECK(s.dtype() == torch::kFloat16 && s.is_contiguous(), "scales must be contiguous fp16 in blocked layout");
    const int rows = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N % kRowsPerBlock == 0, "N must be a multiple of 64");
    TORCH_CHECK(rows >= 1 && rows <= kMaxRows, "rows must be in [1, 64]");
    const int groups = K / 128;
    const int warps_n = N / kRowBlock;
    int split = min(kMaxSplitK, max(1, (kTargetWarps + warps_n - 1) / warps_n));
    const int groups_per_split = (groups + split - 1) / split;
    split = (groups + groups_per_split - 1) / groups_per_split;
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const __half*>(s.data_ptr());
    auto y = torch::empty({rows, N}, x.options());
    auto partial = split == 1 ? torch::Tensor() : torch::empty({split, rows, N}, x.options().dtype(torch::kFloat32));
    void* output = split == 1 ? y.data_ptr() : partial.data_ptr();
    if (rows <= 16) launch_mma<1>(xp, wp, sp, output, rows, N, K, split, groups_per_split, stream);
    else if (rows <= 32) launch_mma<2>(xp, wp, sp, output, rows, N, K, split, groups_per_split, stream);
    else launch_mma<4>(xp, wp, sp, output, rows, N, K, split, groups_per_split, stream);
    if (split > 1) {
        const int elements = rows * N;
        reduce_split_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            elements,
            split);
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_gemv", &ternary_gemv, "ternary 2-bit gemm for <=64 rows (bf16 activations, tensor cores, blocked weight layout)");
    m.def("signed_hadamard", &signed_hadamard, "blockwise signed Walsh-Hadamard transform (block 1024)");
}

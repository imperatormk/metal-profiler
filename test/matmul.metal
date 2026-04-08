#include <metal_stdlib>
using namespace metal;

kernel void matmul_naive(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

kernel void softmax(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint blockDim [[threads_per_threadgroup]]
) {
    // Load and find max
    float val = input[gid];
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce for max
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exp and sum
    float e = exp(val - max_val);
    shared[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared[0];

    output[gid] = e / sum;
}

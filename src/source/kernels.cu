#include "kernels.cuh"

#include <chrono>

#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>

__global__ void
setup_curand(
    curandState* state,
    const unsigned long time
);

__global__ void
kernel_alter(
    int n,
    unsigned char* contents,
    unsigned char min_deviation,
    unsigned char max_deviation,
    curandState* state
);


namespace CUDA::kernels
{

    int call_alter_kernel(
        unsigned char* cpu_contents,
        int contents_size,
        unsigned char min_deviation,
        unsigned char max_deviation
    )
    {
        auto current_time_count = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        int thread_count = 2 << 9;
        int block_count = (contents_size + thread_count - 1) / thread_count;

        curandState* random_states;
        if (cudaMalloc(&random_states, sizeof(curandState) * thread_count) != cudaSuccess)
            return 1;

        unsigned char* gpu_contents;
        if (cudaMallocManaged(&gpu_contents, sizeof(unsigned char) * contents_size) != cudaSuccess)
            return 1;

        printf("Copying memory over to CUDA device...\n");
        cudaMemcpy(
            gpu_contents, // destination
            cpu_contents, // source
            sizeof(unsigned char) * contents_size, // byte count
            cudaMemcpyHostToDevice // copy kind
        );

        printf("Setting up the random states...\n");
        ::setup_curand<<<1, thread_count>>>(random_states, current_time_count);

        printf("Altering the given data...\n");
        ::kernel_alter<<<block_count, thread_count>>>(contents_size, gpu_contents, min_deviation, max_deviation, random_states);
        printf("Waiting on device synchronization...\n");
        cudaDeviceSynchronize();

        printf("Copying memory back to cpu...\n");
        cudaMemcpy(
            cpu_contents,
            gpu_contents,
            sizeof(unsigned char) * contents_size,
            cudaMemcpyDeviceToHost
        );

        cudaFree(gpu_contents);
        return 0;
    }

    

} // namespace CUDA::kernels


__global__ void
setup_curand(
    curandState* state,
    const unsigned long time
)
{
    curand_init(time, threadIdx.x, 0, &state[threadIdx.x]);
}

__global__ void
kernel_alter(
    int n,
    unsigned char* contents,
    unsigned char min_deviation,
    unsigned char max_deviation,
    curandState* state
)
{
    curandState currentRandomState = state[threadIdx.x];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = idx; i < n; i += stride)
    {
        unsigned char deviation = (unsigned char)(min_deviation + curand_uniform(&currentRandomState) * (max_deviation - min_deviation));

        contents[i] += deviation;
    }
}

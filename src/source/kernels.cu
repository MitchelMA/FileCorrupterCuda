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
    unsigned long n,
    unsigned char* contents,
    unsigned char min_deviation,
    unsigned char max_deviation,
    curandState* state
);

__global__ void
kernel_pass(
    unsigned long n,
    unsigned char* contents,
    float chance,
    curandState* state
);

__global__ void
kernel_offset(
    unsigned long n,
    unsigned char* contents,
    int offset_amount
);


namespace CUDA::kernels
{

    int call_alter_kernel(
        unsigned char* cpu_contents,
        unsigned long contents_size,
        unsigned char min_deviation,
        unsigned char max_deviation
    )
    {
        struct cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, 0);

        auto current_time_count = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        int thread_count = device_properties.maxThreadsPerBlock;
        int block_count = (contents_size + thread_count - 1) / thread_count;

        curandState* random_states;
        if (cudaMalloc(&random_states, sizeof(curandState) * thread_count) != cudaSuccess)
            return 1;

        unsigned char* gpu_contents;
        if (cudaMallocManaged(&gpu_contents, sizeof(unsigned char) * contents_size))
        {
            cudaFree(random_states);
            return 1;
        }

        printf("Copying memory from RAM to VRAM\n");
        cudaMemcpy(
            gpu_contents,
            cpu_contents,
            sizeof(unsigned char) * contents_size,
            cudaMemcpyHostToDevice
        );

        printf("Setting up the random states...\n");
        ::setup_curand<<<1, thread_count>>>(random_states, current_time_count);

        printf("Altering the given data...\n");
        ::kernel_alter<<<block_count, thread_count>>>(contents_size, gpu_contents, min_deviation, max_deviation, random_states);

        printf("Waiting on device synchronization...\n");
        cudaDeviceSynchronize();
        printf("Synchronization done\n");

        printf("Copying memory back to RAM from VRAM\n");
        cudaMemcpy(
            cpu_contents,
            gpu_contents,
            sizeof(unsigned char) * contents_size,
            cudaMemcpyDeviceToHost
        );

        cudaFree(random_states);
        cudaFree(gpu_contents);

        return 0;
    }

    int call_pass_kernel(
        unsigned char* cpu_contents,
        unsigned long contents_size,
        float chance
    )
    {
        struct cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, 0);

        auto current_time_count = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        int thread_count = device_properties.maxThreadsPerBlock;
        int block_count = (contents_size + thread_count - 1) / thread_count;

        curandState* random_states;
        if (cudaMalloc(&random_states, sizeof(curandState) * thread_count) != cudaSuccess)
            return 1;

        unsigned char* gpu_contents;
        if (cudaMallocManaged(&gpu_contents, sizeof(unsigned char) * contents_size))
        {
            cudaFree(random_states);
            return 1;
        }

        printf("Copying memory from RAM to VRAM\n");
        cudaMemcpy(
            gpu_contents,
            cpu_contents,
            sizeof(unsigned char) * contents_size,
            cudaMemcpyHostToDevice
        );

        printf("Setting up the random states...\n");
        ::setup_curand<<<1, thread_count>>>(random_states, current_time_count);

        printf("Altering the given data...\n");
        ::kernel_pass<<<block_count, thread_count>>>(contents_size, gpu_contents, chance, random_states);

        printf("Waiting on device synchronization...\n");
        cudaDeviceSynchronize();
        printf("Synchronization done\n");

        printf("Copying memory back to RAM from VRAM\n");
        cudaMemcpy(
            cpu_contents,
            gpu_contents,
            sizeof(unsigned char) * contents_size,
            cudaMemcpyDeviceToHost
        );

        cudaFree(random_states);
        cudaFree(gpu_contents);

        return 0;
    }

    int call_offset_kernel(
        unsigned char* cpu_contents,
        unsigned long contents_size,
        int offset_amount
    )
    {
        struct cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, 0);

        int thread_count = device_properties.maxThreadsPerBlock;
        int block_count = (contents_size + thread_count - 1) / thread_count;

        unsigned char* gpu_contents;
        if (cudaMallocManaged(&gpu_contents, sizeof(unsigned char) * contents_size))
            return 1;

        printf("Copying memory from RAM to VRAM\n");
        cudaMemcpy(
            gpu_contents,
            cpu_contents,
            sizeof(unsigned char) * contents_size,
            cudaMemcpyHostToDevice
        );

        printf("Altering the given data with given offset of `%d`...\n", offset_amount);
        ::kernel_offset<<<block_count, thread_count>>>(contents_size, gpu_contents, offset_amount);

        printf("Waiting on device synchronization...\n");
        cudaDeviceSynchronize();
        printf("Synchronization done\n");

        printf("Copying memory back to RAM from VRAM\n");
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
    curand_init(time, threadIdx.x, 0, state + threadIdx.x);
}

__global__ void
kernel_alter(
    unsigned long n,
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

__global__ void
kernel_pass(
    unsigned long n,
    unsigned char* contents,
    float chance,
    curandState* state
)
{
    curandState currentRandomState = state[threadIdx.x];

    unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long stride = blockDim.x * gridDim.x;

    for (unsigned long i = idx; i < n; i += stride)
    {
        float random_value = curand_uniform(&currentRandomState) * 100.f;
        if (random_value <= chance)
            contents[i] = (unsigned char)(curand_uniform(&currentRandomState) * 255.f);
    }
}

__global__ void
kernel_offset(
    unsigned long n,
    unsigned char* contents,
    int offset_amount
)
{
    unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long stride = blockDim.x * gridDim.x;

    for (unsigned long i = idx; i < n; i += stride)
        contents[i] += offset_amount;
}

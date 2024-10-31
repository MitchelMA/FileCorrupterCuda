#include "device_info.cuh"

namespace CUDA::info
{

    int
    get_device_count()
    {
        int num_gpus;
        cudaGetDeviceCount(&num_gpus);
        return num_gpus;
    }

    int 
    get_device_clockrate(
        int device_index
    )
    {
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device_index);
        return properties.clockRate;
    }

    std::string
    get_device_name(
        int device_index
    )
    {
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device_index);
        return properties.name;
    }
} // namespace CUDA::info

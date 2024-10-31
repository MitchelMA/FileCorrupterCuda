#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

namespace CUDA::kernels
{

    int call_alter_kernel(
        unsigned char* cpu_contents,
        int contents_size,
        unsigned char min_deviation,
        unsigned char max_deviation
    );

} // namespace CUDA::kernels


#endif // !CUDA_KERNELS_CUH


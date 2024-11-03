#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

namespace CUDA::kernels
{

    int call_alter_kernel(
        unsigned char* cpu_contents,
        unsigned long contents_size,
        unsigned char min_deviation,
        unsigned char max_deviation
    );

    int call_pass_kernel(
        unsigned char* cpu_contents,
        unsigned long contents_size,
        float chance
    );

    int call_offset_kernel(
        unsigned char* cpu_contents,
        unsigned long contents_size,
        int offset_amount
    );

} // namespace CUDA::kernels


#endif // !CUDA_KERNELS_CUH


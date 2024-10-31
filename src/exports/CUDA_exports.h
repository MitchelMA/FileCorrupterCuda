#ifndef CUDA_EXPORTS_H
#define CUDA_EXPORTS_H

#define DLL_ENTRY extern "C"
#define STD_DLL_ENTRY(RET_TYPE) DLL_ENTRY RET_TYPE

namespace info
{

    STD_DLL_ENTRY(int)
    get_device_count();

    STD_DLL_ENTRY(bool)
    get_device_name(int device_index, char* name, int* name_len);

} // namespace info

namespace kernels
{
    STD_DLL_ENTRY(int)
    alter_kernel(unsigned char* contents, unsigned long element_count, int min_deviation, int max_deviation);

    STD_DLL_ENTRY(int)
    pass_kernel(unsigned char* contents, unsigned long element_count, float chance);
}

#endif // !CUDA_EXPORTS_H


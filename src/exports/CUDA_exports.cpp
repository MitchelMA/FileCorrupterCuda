#include "CUDA_exports.h"

#include "device_info.cuh"
#include "kernels.cuh"

#include <cstring>

namespace info
{

    STD_DLL_ENTRY(int)
    get_device_count()
    {
        return CUDA::info::get_device_count();
    }

    STD_DLL_ENTRY(bool)
    get_device_name(
        int device_index,
        char* name,
        int* name_len
    )
    {
        if (get_device_count() <= 0)
            return false;

        std::string device_name = CUDA::info::get_device_name(device_index);
        std::strncpy(name, device_name.c_str(), device_name.size());

        return true;
    }

} // namespace info

namespace kernels
{

    STD_DLL_ENTRY(int)
    alter_kernel(
        unsigned char* contents,
        unsigned long element_count,
        int min_deviation,
        int max_deviation
    )
    {
        auto success = CUDA::kernels::call_alter_kernel(
            contents,
            element_count,
            min_deviation,
            max_deviation
        );

        if (success == 1)
            return 0;

        return element_count;
    }

    STD_DLL_ENTRY(int)
    pass_kernel(
        unsigned char* contents,
        unsigned long element_count,
        float chance
    )
    {
        auto success = CUDA::kernels::call_pass_kernel(
            contents,
            element_count,
            chance
        );

        if (success == 1)
            return 0;

        return element_count;
    }

} // namespace kernels


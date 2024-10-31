#ifndef DEVICE_INFO_CUH
#define DEVICE_INFO_CUH

#include <string>

namespace CUDA::info
{

    int get_device_count();
    int get_device_clockrate(int device_index);
    std::string get_device_name(int device_index);

} // namespace CUDA::info

#endif // !DEVICE_INFO_CUH

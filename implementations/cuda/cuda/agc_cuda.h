#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
#define AGC_CUDA_PUBLIC __declspec(dllimport)
#else
#define AGC_CUDA_PUBLIC
#endif
#ifdef __cplusplus
extern "C" {
#endif
// family: HE14=0, HE8=1, CLAHE=2, plateau=3, linear=4, manual=5, auto-bright=6.
// One context per concurrent caller; dimensions/configuration are fixed at creation.
// Host/device input must be contiguous uint16, range 0..16383. Output is uint8.
AGC_CUDA_PUBLIC void* agc_cuda_create(int width,int height,int family,double tail,double clip,int grid,int plateau,int midpoint,double contrast,double brightness);
AGC_CUDA_PUBLIC void agc_cuda_destroy(void* context);
AGC_CUDA_PUBLIC int agc_cuda_render(void* context,const uint16_t* input,uint8_t* output,float* gpu_interval_ms);
AGC_CUDA_PUBLIC int agc_cuda_upload(void* context,const uint16_t* input);
AGC_CUDA_PUBLIC int agc_cuda_device_time(void* context,float* ms);
AGC_CUDA_PUBLIC int agc_cuda_device_time_many(void* context,int repeats,float* amortized_ms);
AGC_CUDA_PUBLIC int agc_cuda_batch_render(void** contexts,const uint16_t** inputs,uint8_t** outputs,int count);
AGC_CUDA_PUBLIC size_t agc_cuda_workspace(void* context);
AGC_CUDA_PUBLIC const char* agc_cuda_error(void);
AGC_CUDA_PUBLIC const char* agc_cuda_info(void);
// Borrowed pointers/stream remain valid until destruction. Do not free them.
// Populate the input on the returned stream, launch, then consume output on that
// same stream (or establish an explicit CUDA event dependency).
AGC_CUDA_PUBLIC void* agc_cuda_input_device(void* context);
AGC_CUDA_PUBLIC void* agc_cuda_output_device(void* context);
AGC_CUDA_PUBLIC void* agc_cuda_stream(void* context);
AGC_CUDA_PUBLIC int agc_cuda_launch_async(void* context);
#ifdef __cplusplus
}
#endif

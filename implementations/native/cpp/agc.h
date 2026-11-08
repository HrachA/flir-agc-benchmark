#pragma once
#include <stdint.h>
#if defined(_WIN32)
#define AGC_PUBLIC __declspec(dllimport)
#else
#define AGC_PUBLIC
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Family: 0=HE14,1=HE8,2=CLAHE,3=plateau,4=linear,5=manual,6=auto-bright.
// Input/output must be non-overlapping, contiguous W*H buffers. Input is 0..16383.
// One renderer handle per concurrent caller. Set global OpenCV threads before use.
AGC_PUBLIC void* agc_create(int family,double tail,double clip,int grid,int plateau,
                           int midpoint,double contrast,double brightness,int fast);
AGC_PUBLIC int agc_render(void* renderer,const uint16_t* input,int width,int height,uint8_t* output);
AGC_PUBLIC void agc_destroy(void* renderer);
AGC_PUBLIC const char* agc_last_error(void);
AGC_PUBLIC void agc_set_threads(int threads);
AGC_PUBLIC const char* agc_opencv_info(void);
AGC_PUBLIC const char* agc_compiler(void);
#ifdef __cplusplus
}
#endif

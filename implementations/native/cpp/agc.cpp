// OpenCV HE/CLAHE + fused native FLIR-inspired global mappings.
// No firmware-equivalence claim. Default rounding mode: nearest, ties to even.
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)
namespace {
constexpr int B = 16384;
thread_local std::string last_error;
struct Renderer {
    int family, grid, plateau, midpoint;
    double tail, clip, contrast, brightness;
    bool fast;
    std::array<uint32_t, B*4> scratch{};
    std::array<uint32_t, B> hist{};
    std::array<uint8_t, B> lut{};
    cv::Mat base;
    cv::Ptr<cv::CLAHE> clahe;
    Renderer(int f,double t,double c,int g,int p,int m,double co,double br,bool fa):
        family(f),grid(g),plateau(p),midpoint(m),tail(t),clip(c),contrast(co),brightness(br),fast(fa) {
        if(f<0 || f>6 || t<0 || t>=50 || co<=0 || (f==2 && (g<1 || c<=0)) || (f==3 && p<1))
            throw std::invalid_argument("Invalid renderer parameters");
        if(f==2)clahe=cv::createCLAHE(clip,cv::Size(grid,grid));
    }
    static uint8_t round8(double x) {
        if(x<=0)return 0;if(x>=255)return 255;
        return static_cast<uint8_t>(cvRound(x));
    }
    void histogram(const uint16_t* src,size_t n) {
        std::fill(scratch.begin(),scratch.end(),0);
        auto* a=scratch.data();auto* b=a+B;auto* c=b+B;auto* d=c+B;
        size_t i=0;
        for(;i+3<n;i+=4) {
            if((src[i]|src[i+1]|src[i+2]|src[i+3])>=B)throw std::invalid_argument("Input exceeds 14-bit range");
            ++a[src[i]];++b[src[i+1]];++c[src[i+2]];++d[src[i+3]];
        }
        for(;i<n;++i){if(src[i]>=B)throw std::invalid_argument("Input exceeds 14-bit range");++a[src[i]];}
        for(int k=0;k<B;++k)hist[k]=a[k]+b[k]+c[k]+d[k];
    }
    double percentile(size_t n,double percent) const {
        double rank=static_cast<double>(n-1)*percent/100.0;
        uint64_t low=static_cast<uint64_t>(std::floor(rank)),high=static_cast<uint64_t>(std::ceil(rank));
        uint64_t acc=0;int a=-1,b=0;
        for(int k=0;k<B;++k){acc+=hist[k];if(a<0 && acc>low)a=k;if(acc>high){b=k;break;}}
        return a+(b-a)*(rank-std::floor(rank));
    }
    void map(const uint16_t* src,uint8_t* dst,size_t n) const {
        size_t i=0;
        for(;i+7<n;i+=8){
            dst[i]=lut[src[i]];dst[i+1]=lut[src[i+1]];dst[i+2]=lut[src[i+2]];dst[i+3]=lut[src[i+3]];
            dst[i+4]=lut[src[i+4]];dst[i+5]=lut[src[i+5]];dst[i+6]=lut[src[i+6]];dst[i+7]=lut[src[i+7]];
        }
        for(;i<n;++i)dst[i]=lut[src[i]];
    }
    void cdf_lut() {
        uint64_t total=0,first=0;int occupied=0;
        for(auto count:hist){total+=count;if(count){if(!occupied)first=count;++occupied;}}
        if(occupied<2){lut.fill(0);return;}
        double scale=255.0/static_cast<double>(total-first);uint64_t acc=0;
        for(int k=0;k<B;++k){acc+=hist[k];lut[k]=round8((static_cast<double>(acc)-static_cast<double>(first))*scale);}
    }
    void render(const uint16_t* src,int w,int h,uint8_t* dst) {
        if(!src || !dst || w<=0 || h<=0 || static_cast<uint64_t>(w)*h>UINT32_MAX)
            throw std::invalid_argument("Invalid image/buffer dimensions");
        size_t n=static_cast<size_t>(w)*h;
        cv::Mat input(h,w,CV_16UC1,const_cast<uint16_t*>(src));
        cv::Mat output(h,w,CV_8UC1,dst);
        if(family==5) {
            if(fast){double gain=contrast/64.0;input.convertTo(output,CV_8U,gain,127-brightness*gain);return;}
            // Generic explicit float32 operation order matches the Python reference.
            for(int k=0;k<B;++k){float x=static_cast<float>(k)-static_cast<float>(brightness);
                x=x*static_cast<float>(contrast);x=x/64.0f;x=x+127.0f;lut[k]=round8(x);}
            // Check range before the indexed LUT application.
            for(size_t i=0;i<n;++i)if(src[i]>=B)throw std::invalid_argument("Input exceeds 14-bit range");
            map(src,dst,n);return;
        }
        if(family==6) {
            uint64_t sum=0;
            for(size_t i=0;i<n;++i){if(src[i]>=B)throw std::invalid_argument("Input exceeds 14-bit range");sum+=src[i];}
            double mean=static_cast<double>(sum)/n;
            if(fast){double gain=contrast/64.0;input.convertTo(output,CV_8U,gain,127-mean*gain);return;}
            for(int k=0;k<B;++k){double x=(static_cast<double>(k)-mean)*contrast/64.0+127.0;lut[k]=round8(x);}
            map(src,dst,n);return;
        }
        histogram(src,n);
        if(family==0){cdf_lut();map(src,dst,n);return;}
        double lo=percentile(n,tail),hi=percentile(n,100-tail);
        if(family==3){
            for(int k=0;k<B;++k)hist[k]=(k<lo || k>hi)?0:std::min(hist[k],static_cast<uint32_t>(plateau));
            cdf_lut();map(src,dst,n);return;
        }
        if(family==4){
            if(hi<=lo){output.setTo(midpoint);return;}
            if(fast){double gain=255.0/(hi-lo);input.convertTo(output,CV_8U,gain,midpoint-(lo+hi)/2*gain);return;}
            float mid=static_cast<float>((lo+hi)/2),span=static_cast<float>(hi-lo);
            for(int k=0;k<B;++k){float x=static_cast<float>(k)-mid;x=x*255.0f;x=x/span;x=x+static_cast<float>(midpoint);lut[k]=round8(x);}
            map(src,dst,n);return;
        }
        base.create(h,w,CV_8UC1);
        if(hi<=lo)base.setTo(0);
        else if(fast){double gain=255.0/(hi-lo);input.convertTo(base,CV_8U,gain,-lo*gain);}
        else {
            float low=static_cast<float>(lo),scale=static_cast<float>(255.0/(hi-lo));
            for(int k=0;k<B;++k){float x=static_cast<float>(k)-low;x=x*scale;lut[k]=round8(x);}
            map(src,base.ptr<uint8_t>(),n);
        }
        if(family==1)cv::equalizeHist(base,output);
        else if(family==2)clahe->apply(base,output);
        else throw std::invalid_argument("Unknown family");
    }
};
}
API void* agc_create(int family,double tail,double clip,int grid,int plateau,int midpoint,double contrast,double brightness,int fast) {
    try{return new Renderer(family,tail,clip,grid,plateau,midpoint,contrast,brightness,fast!=0);}
    catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API int agc_render(void* handle,const uint16_t* src,int width,int height,uint8_t* dst){
    try{if(!handle)throw std::invalid_argument("Null renderer");static_cast<Renderer*>(handle)->render(src,width,height,dst);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
API void agc_destroy(void* handle){delete static_cast<Renderer*>(handle);}
API const char* agc_last_error(){return last_error.c_str();}
API void agc_set_threads(int n){cv::setNumThreads(n);cv::setUseOptimized(true);cv::ocl::setUseOpenCL(false);}
API const char* agc_opencv_info(){return cv::getBuildInformation().c_str();}
API const char* agc_compiler(){
#ifdef _MSC_VER
    return "MSVC " STR(_MSC_VER) "; C++17; /O2 /Oi /fp:precise; AVX2 when enabled by CMake";
#else
    return __VERSION__;
#endif
}

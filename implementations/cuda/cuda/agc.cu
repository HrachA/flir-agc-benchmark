// Native CUDA reference-order AGC; OpenCV-compatible 8-bit CLAHE convention.
// See THIRD_PARTY_NOTICES.txt for OpenCV algorithm attribution/license.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
namespace {
constexpr int B=16384;
thread_local std::string error_text;
inline void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
struct Params {double lo,hi,scale,mean;int first,total,first_index;};
__device__ int search(const int* cdf,int bins,int target){int a=0,b=bins-1;while(a<b){int mid=(a+b)/2;if(cdf[mid]<target)a=mid+1;else b=mid;}return a;}
__device__ double percentile(const int* cdf,int n,double p){
    double rank=__ddiv_rn(__dmul_rn(static_cast<double>(n-1),p),100.0);
    double low=floor(rank);int a=search(cdf,B,static_cast<int>(low)+1),b=search(cdf,B,static_cast<int>(ceil(rank))+1);
    return __dadd_rn(static_cast<double>(a),__dmul_rn(static_cast<double>(b-a),__dsub_rn(rank,low)));
}
__device__ uint8_t byte_d(double x){return static_cast<uint8_t>(__double2int_rn(fmin(255.0,fmax(0.0,x))));}
__device__ uint8_t byte_f(float x){return static_cast<uint8_t>(__float2int_rn(fminf(255.0f,fmaxf(0.0f,x))));}
__global__ void statistics(const int* cdf,int bins,int n,double tail,Params* p,int quantiles){
    if(threadIdx.x==0){int first=search(cdf,bins,1);p->first_index=first;p->first=cdf[first];p->total=cdf[bins-1];
        p->scale=p->total>p->first?255.0/static_cast<double>(p->total-p->first):0;
        if(quantiles){p->lo=percentile(cdf,n,tail);p->hi=percentile(cdf,n,100-tail);}}
}
__global__ void clip_hist(const int* hist,int* clipped,const Params* p,int cap){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<B)clipped[k]=(k<p->lo || k>p->hi)?0:min(hist[k],cap);
}
__global__ void cdf_map(const uint16_t* src,uint8_t* out,int n,const int* cdf,const Params* p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=byte_d(__dmul_rn(static_cast<double>(cdf[src[i]]-p->first),p->scale));
}
__global__ void stretch(const uint16_t* src,uint8_t* out,int n,const Params* p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
        float gain=static_cast<float>(255.0/(p->hi-p->lo));
        float x=__fsub_rn(static_cast<float>(src[i]),static_cast<float>(p->lo));
        out[i]=p->hi>p->lo?byte_f(__fmul_rn(x,gain)):0;}
}
__global__ void he8_map(const uint8_t* src,uint8_t* out,int n,const int* cdf,const Params* p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
        float scale=p->total>p->first?255.0f/static_cast<float>(p->total-p->first):0.0f;
        out[i]=p->total==p->first?static_cast<uint8_t>(p->first_index):byte_f(__fmul_rn(static_cast<float>(cdf[src[i]]-p->first),scale));}
}
__global__ void linear_map(const uint16_t* src,uint8_t* out,int n,const Params* p,int midpoint){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
        float mid=static_cast<float>((p->lo+p->hi)/2),span=static_cast<float>(p->hi-p->lo);
        float x=__fsub_rn(static_cast<float>(src[i]),mid);x=__fmul_rn(x,255.f);x=__fdiv_rn(x,span);x=__fadd_rn(x,static_cast<float>(midpoint));
        out[i]=p->hi>p->lo?byte_f(x):static_cast<uint8_t>(midpoint);}
}
__global__ void manual_map(const uint16_t* src,uint8_t* out,int n,float contrast,float brightness){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=__fsub_rn(static_cast<float>(src[i]),brightness);
        x=__fmul_rn(x,contrast);x=__fdiv_rn(x,64.f);out[i]=byte_f(__fadd_rn(x,127.f));}
}
struct U64 {__host__ __device__ unsigned long long operator()(uint16_t x)const{return x;}};
__global__ void auto_map(const uint16_t* src,uint8_t* out,int n,const unsigned long long* sum,double contrast){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){double mean=__ddiv_rn(static_cast<double>(*sum),static_cast<double>(n));
        double x=__dsub_rn(static_cast<double>(src[i]),mean);x=__dmul_rn(x,contrast);x=__ddiv_rn(x,64.0);out[i]=byte_d(__dadd_rn(x,127.0));}
}
__device__ int reflect101(int x,int n){if(n<=1)return 0;int p=2*(n-1);x%=p;return x<n?x:p-x;}
__global__ void tile_luts(const uint8_t* base,uint8_t* luts,int w,int h,int grid,int tw,int th,int limit){
    __shared__ int hist[8][256];
    __shared__ int clipped;
    typedef cub::BlockReduce<int,256> Reduce;
    typedef cub::BlockScan<int,256> Scan;
    __shared__ union {typename Reduce::TempStorage reduce;typename Scan::TempStorage scan;} temp;
    int tid=threadIdx.x,warp=tid/32;
    for(int k=tid;k<8*256;k+=256)reinterpret_cast<int*>(hist)[k]=0;
    __syncthreads();
    int tx=blockIdx.x%grid,ty=blockIdx.x/grid;
    for(int j=tid;j<tw*th;j+=256){int x=reflect101(tx*tw+j%tw,w),y=reflect101(ty*th+j/tw,h);int v=base[y*w+x];
        atomicAdd(&hist[warp][v],1);}
    __syncthreads();
    int count=0;for(int k=0;k<8;++k)count+=hist[k][tid];
    int excess=max(count-limit,0);count=min(count,limit);
    int total=Reduce(temp.reduce).Sum(excess);
    if(tid==0)clipped=total;
    __syncthreads();
    int batch=clipped/256,residual=clipped%256,step=residual?max(256/residual,1):1;
    count+=batch+((residual && tid%step==0 && tid/step<residual)?1:0);
    __syncthreads();
    int cdf;Scan(temp.scan).InclusiveSum(count,cdf);
    float scale=255.f/static_cast<float>(tw*th);
    luts[blockIdx.x*256+tid]=byte_f(__fmul_rn(static_cast<float>(cdf),scale));
}
__global__ void interpolate(const uint8_t* base,const uint8_t* luts,uint8_t* out,int w,int h,int grid,int tw,int th){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
    int x=i%w,y=i/w,v=base[i];float txf=__fsub_rn(__fmul_rn(static_cast<float>(x),1.f/tw),.5f);
    float tyf=__fsub_rn(__fmul_rn(static_cast<float>(y),1.f/th),.5f);
    int tx1=static_cast<int>(floorf(txf)),tx2=tx1+1,ty1=static_cast<int>(floorf(tyf)),ty2=ty1+1;
    float xa=__fsub_rn(txf,static_cast<float>(tx1)),ya=__fsub_rn(tyf,static_cast<float>(ty1));
    float xa1=__fsub_rn(1.f,xa),ya1=__fsub_rn(1.f,ya);
    tx1=max(tx1,0);tx2=min(tx2,grid-1);ty1=max(ty1,0);ty2=min(ty2,grid-1);
    float a=__fadd_rn(__fmul_rn(static_cast<float>(luts[(ty1*grid+tx1)*256+v]),xa1),__fmul_rn(static_cast<float>(luts[(ty1*grid+tx2)*256+v]),xa));
    float b=__fadd_rn(__fmul_rn(static_cast<float>(luts[(ty2*grid+tx1)*256+v]),xa1),__fmul_rn(static_cast<float>(luts[(ty2*grid+tx2)*256+v]),xa));
    out[i]=byte_f(__fadd_rn(__fmul_rn(a,ya1),__fmul_rn(b,ya)));
}
struct Context {
    int w,h,n,family,grid,plateau,midpoint,tw,th,limit;double tail,clip,contrast,brightness;
    cudaStream_t stream{};cudaGraph_t graph{},transfer_graph{};cudaGraphExec_t exec{},transfer_exec{};
    cudaEvent_t start{},stop{},rt_start{},rt_stop{};
    uint16_t* raw{};uint8_t *base{},*output{},*tiles{},*host_output{};uint16_t* host_input{};
    int *hist{},*cdf{},*clipped{},*cdf_clipped{};Params* params{};unsigned long long* sum{};
    void* temp{};size_t temp_bytes{},device_bytes{};
    template<class T>void alloc(T*& p,size_t bytes){check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));device_bytes+=bytes;}
    Context(int width,int height,int f,double t,double c,int g,int p,int m,double co,double br):
        w(width),h(height),n(width*height),family(f),grid(g),plateau(p),midpoint(m),tail(t),clip(c),contrast(co),brightness(br){
        if(w<=0 || h<=0 || family<0 || family>6 || grid<1 || tail<0 || tail>=50)throw std::invalid_argument("Invalid CUDA parameters");
        bool padded=(w%grid || h%grid);tw=(w+(padded?grid-w%grid:0))/grid;th=(h+(padded?grid-h%grid:0))/grid;
        limit=std::max(static_cast<int>(clip*tw*th/256),1);
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        alloc(raw,static_cast<size_t>(n)*2);alloc(output,n);
        if(family!=5 && family!=6){alloc(hist,B*sizeof(int));alloc(cdf,B*sizeof(int));alloc(params,sizeof(Params));}
        if(family==3){alloc(clipped,B*sizeof(int));alloc(cdf_clipped,B*sizeof(int));}
        if(family==1 || family==2)alloc(base,n);
        if(family==2)alloc(tiles,grid*grid*256);
        if(family==6)alloc(sum,sizeof(unsigned long long));
        check(cudaMallocHost(reinterpret_cast<void**>(&host_input),static_cast<size_t>(n)*2));check(cudaMallocHost(reinterpret_cast<void**>(&host_output),n));
        size_t a=0,b=0,cbytes=0,d=0;
        if(family!=5 && family!=6){
            check(cub::DeviceHistogram::HistogramEven(nullptr,a,raw,hist,B+1,0,16384,n,stream));
            check(cub::DeviceScan::InclusiveSum(nullptr,cbytes,hist,cdf,B,stream));}
        if(family==1)check(cub::DeviceHistogram::HistogramEven(nullptr,b,base,hist,257,0,256,n,stream));
        cub::TransformInputIterator<unsigned long long,U64,const uint16_t*> values(raw,U64{});
        if(family==6)check(cub::DeviceReduce::Sum(nullptr,d,values,sum,n,stream));
        temp_bytes=std::max(std::max(a,b),std::max(cbytes,d));if(temp_bytes)alloc(temp,temp_bytes);
        check(cudaEventCreate(&start));check(cudaEventCreate(&stop));check(cudaEventCreate(&rt_start));check(cudaEventCreate(&rt_stop));
        check(cudaMemsetAsync(raw,0,static_cast<size_t>(n)*2,stream));enqueue();check(cudaStreamSynchronize(stream));
        capture(false);capture(true);
    }
    void enqueue(){
        int blocks=(n+255)/256;
        if(family==5){manual_map<<<blocks,256,0,stream>>>(raw,output,n,static_cast<float>(contrast),static_cast<float>(brightness));return;}
        if(family==6){cub::TransformInputIterator<unsigned long long,U64,const uint16_t*> values(raw,U64{});
            check(cub::DeviceReduce::Sum(temp,temp_bytes,values,sum,n,stream));auto_map<<<blocks,256,0,stream>>>(raw,output,n,sum,contrast);return;}
        check(cub::DeviceHistogram::HistogramEven(temp,temp_bytes,raw,hist,B+1,0,16384,n,stream));
        check(cub::DeviceScan::InclusiveSum(temp,temp_bytes,hist,cdf,B,stream));
        statistics<<<1,32,0,stream>>>(cdf,B,n,tail,params,family!=0);
        if(family==0){cdf_map<<<blocks,256,0,stream>>>(raw,output,n,cdf,params);return;}
        if(family==3){clip_hist<<<B/256,256,0,stream>>>(hist,clipped,params,plateau);
            check(cub::DeviceScan::InclusiveSum(temp,temp_bytes,clipped,cdf_clipped,B,stream));
            statistics<<<1,32,0,stream>>>(cdf_clipped,B,n,tail,params,0);
            cdf_map<<<blocks,256,0,stream>>>(raw,output,n,cdf_clipped,params);return;}
        if(family==4){linear_map<<<blocks,256,0,stream>>>(raw,output,n,params,midpoint);return;}
        stretch<<<blocks,256,0,stream>>>(raw,base,n,params);
        if(family==1){check(cub::DeviceHistogram::HistogramEven(temp,temp_bytes,base,hist,257,0,256,n,stream));
            check(cub::DeviceScan::InclusiveSum(temp,temp_bytes,hist,cdf,256,stream));
            statistics<<<1,32,0,stream>>>(cdf,256,n,tail,params,0);he8_map<<<blocks,256,0,stream>>>(base,output,n,cdf,params);return;}
        tile_luts<<<grid*grid,256,0,stream>>>(base,tiles,w,h,grid,tw,th,limit);
        interpolate<<<blocks,256,0,stream>>>(base,tiles,output,w,h,grid,tw,th);
    }
    void capture(bool transfer){
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if(transfer)check(cudaMemcpyAsync(raw,host_input,static_cast<size_t>(n)*2,cudaMemcpyHostToDevice,stream));
        enqueue();
        if(transfer)check(cudaMemcpyAsync(host_output,output,n,cudaMemcpyDeviceToHost,stream));
        cudaGraph_t* target=transfer?&transfer_graph:&graph;check(cudaStreamEndCapture(stream,target));
        check(cudaGraphInstantiate(transfer?&transfer_exec:&exec,*target,nullptr,nullptr,0));
    }
    ~Context(){
        cudaStreamSynchronize(stream);if(exec)cudaGraphExecDestroy(exec);if(transfer_exec)cudaGraphExecDestroy(transfer_exec);
        if(graph)cudaGraphDestroy(graph);if(transfer_graph)cudaGraphDestroy(transfer_graph);
        void* buffers[]={raw,base,output,tiles,hist,cdf,clipped,cdf_clipped,params,sum,temp};
        for(void* p:buffers)if(p)cudaFree(p);
        if(host_input)cudaFreeHost(host_input);if(host_output)cudaFreeHost(host_output);
        for(auto event:{start,stop,rt_start,rt_stop})if(event)cudaEventDestroy(event);if(stream)cudaStreamDestroy(stream);
    }
};
}
API const char* agc_cuda_error(){return error_text.c_str();}
API void* agc_cuda_create(int w,int h,int family,double tail,double clip,int grid,int plateau,int midpoint,double contrast,double brightness){
    try{return new Context(w,h,family,tail,clip,grid,plateau,midpoint,contrast,brightness);}catch(const std::exception& e){error_text=e.what();return nullptr;}}
API void agc_cuda_destroy(void* p){delete static_cast<Context*>(p);}
API int agc_cuda_render(void* p,const uint16_t* input,uint8_t* output,float* gpu_ms){
    try{auto& c=*static_cast<Context*>(p);std::memcpy(c.host_input,input,static_cast<size_t>(c.n)*2);
        if(gpu_ms)check(cudaEventRecord(c.rt_start,c.stream));
        check(cudaGraphLaunch(c.transfer_exec,c.stream));
        if(gpu_ms)check(cudaEventRecord(c.rt_stop,c.stream));
        check(cudaStreamSynchronize(c.stream));
        if(gpu_ms)check(cudaEventElapsedTime(gpu_ms,c.rt_start,c.rt_stop));std::memcpy(output,c.host_output,c.n);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}}
API int agc_cuda_upload(void* p,const uint16_t* input){
    try{auto& c=*static_cast<Context*>(p);std::memcpy(c.host_input,input,static_cast<size_t>(c.n)*2);
        check(cudaMemcpyAsync(c.raw,c.host_input,static_cast<size_t>(c.n)*2,cudaMemcpyHostToDevice,c.stream));check(cudaStreamSynchronize(c.stream));return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}}
API int agc_cuda_device_time(void* p,float* ms){
    try{auto& c=*static_cast<Context*>(p);check(cudaEventRecord(c.start,c.stream));check(cudaGraphLaunch(c.exec,c.stream));
        check(cudaEventRecord(c.stop,c.stream));check(cudaStreamSynchronize(c.stream));
        check(cudaEventElapsedTime(ms,c.start,c.stop));return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}}
API int agc_cuda_device_time_many(void* p,int repeats,float* ms){
    try{if(repeats<1)throw std::invalid_argument("repeats must be positive");auto& c=*static_cast<Context*>(p);
        check(cudaEventRecord(c.start,c.stream));for(int i=0;i<repeats;++i)check(cudaGraphLaunch(c.exec,c.stream));
        check(cudaEventRecord(c.stop,c.stream));check(cudaStreamSynchronize(c.stream));
        check(cudaEventElapsedTime(ms,c.start,c.stop));*ms/=repeats;return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}}
API int agc_cuda_batch_render(void** handles,const uint16_t** inputs,uint8_t** outputs,int count){
    try{if(count<1)throw std::invalid_argument("Empty batch");
        for(int i=0;i<count;++i){auto& c=*static_cast<Context*>(handles[i]);std::memcpy(c.host_input,inputs[i],static_cast<size_t>(c.n)*2);}
        for(int i=0;i<count;++i){auto& c=*static_cast<Context*>(handles[i]);check(cudaGraphLaunch(c.transfer_exec,c.stream));}
        for(int i=0;i<count;++i){auto& c=*static_cast<Context*>(handles[i]);check(cudaStreamSynchronize(c.stream));std::memcpy(outputs[i],c.host_output,c.n);}return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}}
API void* agc_cuda_input_device(void* p){return static_cast<Context*>(p)->raw;}
API void* agc_cuda_output_device(void* p){return static_cast<Context*>(p)->output;}
API void* agc_cuda_stream(void* p){return static_cast<Context*>(p)->stream;}
API int agc_cuda_launch_async(void* p){
    try{auto& c=*static_cast<Context*>(p);check(cudaGraphLaunch(c.exec,c.stream));return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}}
API size_t agc_cuda_workspace(void* p){return static_cast<Context*>(p)->device_bytes;}
API const char* agc_cuda_info(){
    static std::string info;try{cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,0));int driver=0,runtime=0;cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime);
        std::ostringstream s;s<<"{\"gpu\":\""<<prop.name<<"\",\"compute_capability\":\""<<prop.major<<'.'<<prop.minor
        <<"\",\"sm_count\":"<<prop.multiProcessorCount<<",\"global_memory_bytes\":"<<prop.totalGlobalMem
        <<",\"driver_api_version\":"<<driver<<",\"runtime_version\":"<<runtime<<",\"nvcc\":\""<<__CUDACC_VER_MAJOR__<<'.'<<__CUDACC_VER_MINOR__<<"\"}";info=s.str();return info.c_str();
    }catch(const std::exception& e){error_text=e.what();return nullptr;}}

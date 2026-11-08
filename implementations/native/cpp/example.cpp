#include "agc.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"Usage: agc_example input.tiff output.png\n";return 2;}
    cv::Mat raw=cv::imread(argv[1],cv::IMREAD_UNCHANGED);
    if(raw.empty() || raw.type()!=CV_16UC1){std::cerr<<"Expected a single-channel uint16 TIFF\n";return 2;}
    if(!raw.isContinuous())raw=raw.clone();
    cv::Mat output(raw.rows,raw.cols,CV_8UC1);
    agc_set_threads(1);
    void* renderer=agc_create(3,0.1,2.0,8,1000,127,1.0,8192.0,0);
    if(!renderer){std::cerr<<agc_last_error()<<'\n';return 1;}
    int rc=agc_render(renderer,raw.ptr<uint16_t>(),raw.cols,raw.rows,output.ptr<uint8_t>());
    agc_destroy(renderer);
    if(rc){std::cerr<<agc_last_error()<<'\n';return 1;}
    if(!cv::imwrite(argv[2],output)){std::cerr<<"Could not save output\n";return 1;}
    return 0;
}

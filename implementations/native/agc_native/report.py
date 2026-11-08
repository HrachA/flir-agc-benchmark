"""Native implementation efficiency report and publication figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .support import read_csv,write_csv

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def work(spec,engine,w=640,h=512):
    b,n,f=16384,w*h,spec['family'];fast='fast' in engine
    passes={'he':2,'he8':4,'plateau':2,'clahe':4,'linear_hist':2,'manual':1 if fast else 2,'auto_bright':2}[f]
    bin_ops={'he':10*b,'he8':(10 if fast else 11)*b+512,'plateau':13*b,
             'clahe':(10 if fast else 11)*b+3*spec.get('grid',0)**2*256,
             'linear_hist':(10 if fast else 11)*b,'manual':0 if fast else b,'auto_bright':0 if fast else b}[f]
    return {'engine':engine,'method':spec['name'],'width':w,'height':h,'pixels':n,
            'conceptual_pixel_passes':passes,'bin_work_proxy':bin_ops,
            'work_proxy':passes*n+bin_ops,'formula':f'{passes}N + {bin_ops}',
            'asymptotic':('O(N)' if fast and f in ['manual','auto_bright'] else 'O(N + B + G*256)' if f=='clahe' else 'O(N + B)'),
            'persistent_global_workspace_bytes':21*b,
            'note':'Coarse pixel/bin visits, not FLOPs or CPU instruction counts; persistent workspace reused.'}

def create_report(out,cfg):
    quality=read_csv(out/'quality_summary.csv');timing=read_csv(out/'latency_summary.csv');parity=read_csv(out/'parity_summary.csv')
    q={(r['engine'],r['split'],r['method']):r for r in quality}
    t={(r['engine'],r['split'],r['method']):r for r in timing}
    p={(r['engine'],r['split'],r['method']):r for r in parity}
    env=json.loads((out/'environment_latency.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    val=[r for r in quality if r['engine']=='python' and r['split']=='val']
    chosen=[]
    for family in dict.fromkeys(s['family'] for s in cfg['methods']):
        chosen.append(max([r for r in val if r['family']==family],key=lambda r:float(r['ssim_mean'])))
    chosen.sort(key=lambda r:float(r['ssim_mean']),reverse=True)
    winner=chosen[0]['method'];winner_test=q[('cpp_exact_1','test',winner)]
    row_summary=[];all_rows=[]
    for r in chosen:
        name=r['method'];base=t[('python','pooled',name)];exact=t[('cpp_exact_1','pooled',name)];fast=t[('cpp_exact_8','pooled',name)]
        row_summary.append([name,f'{float(base["mean_ms"]):.3f}',f'{float(exact["mean_ms"]):.3f}',
                            f'{float(exact["speedup_vs_python"]):.2f}x',f'{float(fast["mean_ms"]):.3f}',
                            f'{float(fast["speedup_vs_python"]):.2f}x',f'{float(winner_test["ssim_mean"]):.4f}' if name==winner else f'{float(q[("cpp_exact_1","test",name)]["ssim_mean"]):.4f}'])
    for s in cfg['methods']:
        n=s['name'];a=t[('python','pooled',n)];b=t[('cpp_exact_1','pooled',n)];c=t[('cpp_fast_1','pooled',n)];d=t[('cpp_fast_8','pooled',n)];e=t[('cpp_exact_8','pooled',n)]
        all_rows.append([n,f'{float(a["mean_ms"]):.3f}',f'{float(b["mean_ms"]):.3f}',f'{float(e["mean_ms"]):.3f}',f'{float(c["mean_ms"]):.3f}',
                         f'{float(d["mean_ms"]):.3f}',f'{float(e["speedup_vs_python"]):.2f}x',
                         f'{float(d["speedup_vs_python"]):.2f}x'])
    total_exact_changed=sum(int(r['different_pixels']) for r in parity if r['engine']=='cpp_exact_1')
    total_fast_changed=sum(int(r['different_pixels']) for r in parity if r['engine']=='cpp_fast_1')
    total_fast_pixels=sum(int(r['pixels']) for r in parity if r['engine']=='cpp_fast_1')
    max_exact=max(int(r['max_abs_difference']) for r in parity if r['engine']=='cpp_exact_1')
    max_fast=max(int(r['max_abs_difference']) for r in parity if r['engine']=='cpp_fast_1')
    delta_max=max(abs(float(q[('cpp_fast_1',split,s['name'])]['ssim_mean'])-float(q[('python',split,s['name'])]['ssim_mean']))
                  for split in ['val','test'] for s in cfg['methods'])
    threads=read_csv(out/'thread_determinism.csv');thread_diff=sum(int(r['different_pixels_threads_1_vs_8']) for r in threads)
    a=t[('python','pooled',winner)];b=t[('cpp_exact_1','pooled',winner)];c=t[('cpp_exact_8','pooled',winner)]
    parity_rows=[]
    for s in cfg['methods']:
        n=s['name'];pv=p[('cpp_fast_1','test',n)];native=q[('cpp_fast_1','test',n)];ref=q[('python','test',n)]
        parity_rows.append([n,pv['max_abs_difference'],f'{float(pv["different_pixel_percent"]):.6f}%',
                            f'{float(ref["ssim_mean"]):.6f}',f'{float(native["ssim_mean"]):.6f}',
                            f'{float(native["ssim_mean"])-float(ref["ssim_mean"]):+.8f}'])
    costs=[work(s,e) for e in ['cpp_exact_1','cpp_exact_8','cpp_fast_1','cpp_fast_8'] for s in cfg['methods']]
    write_csv(out/'complexity_native.csv',costs)
    write_csv(out/'complexity_scaling.csv',[work(s,e,w,h) for w,h in [(160,128),(320,256),(640,512),(1280,1024),(1920,1536)]
                                           for e in ['cpp_exact_1','cpp_exact_8','cpp_fast_1','cpp_fast_8'] for s in cfg['methods']])
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,5),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
    y=np.arange(len(chosen));colors=['#2878B5','#48A36F','#D9791F']
    for i,(engine,label) in enumerate([('python','Original Python'),('cpp_exact_1','C++ exact, 1 thread'),('cpp_exact_8','C++ exact, 8 threads')]):
        values=[float(t[(engine,'pooled',r['method'])]['mean_ms']) for r in chosen]
        ax.barh(y+(i-1)*.23,values,height=.21,color=colors[i],label=label)
    ax.set_yticks(y,[r['method'] for r in chosen]);ax.invert_yaxis();ax.set_xlabel('Mean rendering latency (ms/image)')
    ax.set_title('(a) Same-session CPU timing');ax.legend(frameon=False,fontsize=8)
    for i,(engine,label) in enumerate([('cpp_exact_1','C++ exact, 1 thread'),('cpp_exact_8','C++ exact, 8 threads')]):
        bx.scatter([float(t[(engine,'pooled',r['method'])]['speedup_vs_python']) for r in chosen],y+(i-.5)*.18,
                   color=colors[i+1],marker=['o','s'][i],label=label,s=35)
    bx.axvline(1,color='0.6',lw=1,ls='--');bx.set_xlabel('Speedup over original Python');bx.set_title('(b) Implementation speedup')
    bx.legend(frameon=False,fontsize=8);ax.grid(axis='x',alpha=.2);bx.grid(axis='x',alpha=.2)
    ax.set_axisbelow(True);fig.tight_layout(w_pad=1.5)
    for ext in ['png','pdf']:fig.savefig(figures/f'native_efficiency.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    report=f'''# Native OpenCV/C++ AGC benchmark

## Main finding
The original validation-selected method, **{winner}**, takes **{float(b['mean_ms']):.3f} ms/image** in reference-order C++ at one thread, versus **{float(a['mean_ms']):.3f} ms** for the re-measured Python pipeline: **{float(b['speedup_vs_python']):.2f}x** faster. Its reference-order/8-thread profile takes **{float(c['mean_ms']):.3f} ms** (**{float(c['speedup_vs_python']):.2f}x**). Threads accelerate only eligible OpenCV stages; the custom global histogram and LUT loops remain single-threaded. For plateau HE both thread profiles execute the same single-threaded kernel, so their timing difference reflects run variability rather than parallel acceleration.

The reference-order C++ result reaches test SSIM **{float(winner_test['ssim_mean']):.4f}** and PSNR **{float(winner_test['psnr_db_mean']):.2f} dB**. This is the fastest tested implementation comparison under the recorded settings, not proof of the fastest possible code on all CPUs.

## What was implemented
All 18 settings from the previous experiment are implemented. OpenCV's native **cv::equalizeHist** and **cv::CLAHE** perform HE8 and CLAHE [1,2]. Native 14-bit HE and plateau HE use custom histogram/CDF/LUT code. Percentile bounds, manual mapping, auto-bright and linear-histogram mappings are native C++; the fused profile uses SIMD-dispatched **Mat::convertTo** for point mappings [3]. The detector, GPU and learned renderers are not involved.

Optimizations include four independent histogram accumulators, unrolled LUT application, reusable histogram/LUT/CLAHE workspace, and replacing repeated per-pixel arithmetic with a 16384-entry LUT in the reference-order profile. The fused profile combines scale, offset, clipping and conversion in OpenCV. Custom code is compiled in Release with MSVC /O2 /Oi /fp:precise and AVX2 enabled. Fast-math reassociation is not enabled. Library dispatch, SIMD support and compiler details are recorded.

The Python HE/CLAHE calls already used C++ internally. Speed gains therefore come from the surrounding preprocessing, fewer temporaries, persistent workspace, native loops and threading; they should not be attributed entirely to replacing Python with C++.

## Data and quality checks
The audit retains **{audit['accepted_by_split']['val']} validation and {audit['accepted_by_split']['test']} test pairs**, all 640x512. It excludes {audit['excluded']} TIFFs stored as 8-bit. Source files are unchanged, matched by exact stem and decoded without resizing. Each accepted pair has TIFF/JPEG SHA-256 hashes. The methods are defined in configs/default.json and inherited without test tuning.

Reference-order C++ versus Python: **{total_exact_changed:,} differing pixels**, maximum absolute difference **{max_exact}** gray levels, over all 3295 images x 18 settings. Fused C++: **{total_fast_changed:,} differing pixels ({100*total_fast_changed/total_fast_pixels:.8f}%)**, maximum absolute difference **{max_fast}**. The largest absolute change in a split-level mean SSIM is **{delta_max:.8f}**. Full per-image differences and all reconstruction metrics are archived; changed outputs are scored against the vendor JPEG. Exact matches reuse the hash-validated previous metric values. `--no-cache` recomputes every reference metric.

Reference-order and fused results with 1 and 8 OpenCV threads differ in **{thread_diff} pixels** over every setting on all 128 timing images. Quality scores are computed at one thread; the 8-thread equivalence check is limited to the timing subset and is not represented as a full-dataset proof.

## Validation-selected settings per family
Settings are selected by the original Python validation SSIM, so speed comparisons use identical parameters. All native variants also have independent validation-based selection recorded in selection.csv.

{table(['Method','Python ms','C++ exact/1 ms','Exact/1 speedup','C++ exact/8 ms','Exact/8 speedup','Exact test SSIM'],row_summary)}

## CPU timing protocol
CPU **{env['cpu']}**, {env['physical_cores']} physical cores / {env['logical_processors']} logical processors, {env['ram_gib']:.1f} GiB RAM; {env['platform']}; Python {env['python'].split()[0]}. Packages: {env['packages']}. Native compiler: {env['compiler']}. The official OpenCV C++ SDK is version 5.0.0; the Python wheel is also OpenCV 5.0.0, but they are different binary builds, whose full configuration dumps accompany the results.

The original seeded **64 validation + 64 test frames** are loaded into RAM. There are **640 timed calls per engine/setting** (five repeats), with 20 warm-ups per setting after every engine/thread-profile switch. Engine-block order and frame/method order are seeded and randomized across repeats. Timing includes Python-to-native bridge overhead and a newly allocated output array for every call, as well as all rendering work; internal renderer workspace is reused. There is no hidden precomputed per-image histogram, LUT or output. Baseline Python is re-measured in the same session rather than using historical timings.

Timing excludes image decoding, file I/O, metrics, compilation and one-time renderer construction. GPU/OpenCL are disabled in both OpenCV builds. Python OpenCV uses one thread. Native profiles request one or eight OpenCV threads, without process affinity, fixed clocks or real-time scheduling. Native custom histogram and LUT stages are not parallelized. Thread changes occur outside timed intervals, followed by warm-up. Mean, median, p95, standard deviation, images/s, nanoseconds/pixel and speedups are in latency_summary.csv. Host load and hybrid P/E-core scheduling can affect results. Speedups compare implementations and build/thread choices, not language alone.

## Every fixed setting
{table(['Method','Python ms','C++ exact/1','C++ exact/8','C++ fused/1','C++ fused/8','Exact/8 speedup','Fused/8 speedup'],all_rows)}

## Quantization sensitivity on test
{table(['Method','Fused max error','Changed pixels','Python SSIM','Fused SSIM','SSIM change'],parity_rows)}

Replacing the explicit arithmetic order with one scale/offset conversion changes rounding near quantization thresholds. For histogram-based postprocessing this can affect more than a single gray level after equalization. The reference-order profile is provided when preserving the original rendering is more important than the fastest measured path. Consult the full parity CSV before claiming bit-exact replacement.

## Algorithmic work and memory
Let N=W*H, B=16384 and G be the number of CLAHE tiles. Native global mappings are O(N+B); fused manual and auto-bright are O(N); CLAHE is O(N+B+G*256). The asymptotic order generally matches Python, but native code removes large temporary arrays and moves arithmetic from N pixels to B LUT entries. Four histogram accumulators trade extra small workspace for independent updates.

Each renderer reserves approximately **21B = 344064 bytes** of global histogram/LUT workspace, plus the temporary 8-bit base image for HE8/CLAHE and OpenCV's internal tile buffers. Output memory is N bytes per call. Workspace is allocated once, not counted as a fresh allocation each frame. complexity_native.csv and complexity_scaling.csv provide explicit coarse pixel/bin-visit proxies at five resolutions. Those estimates are not CPU instruction counts or FLOPs; no cycle counters or energy measurements are claimed. Only 640x512 latency is empirically measured.

![Native CPU implementation comparison](figures/native_efficiency.png)

Suggested caption: CPU rendering latency and speedup for the validation-selected setting in each AGC family on an Intel Core i7-13700K. Baseline Python and reference-order C++ requesting one or eight OpenCV threads use the same in-memory image subset and include output allocation. Threading applies only to supported OpenCV stages; the custom global histogram/LUT kernels remain single-threaded. Fused-mode timings and quantization differences are reported separately.

## Limitations and references
These are software implementations inspired by documented FLIR modes [4], not camera firmware. The vendor's exact ADAS JPEG processing settings are undisclosed [5]. Differences from the vendor rendering can remain even with bit-exact agreement to our previous implementation. Results apply to the eligible high-bit-depth subset and tested parameter grid; reconstruction similarity does not establish detection accuracy.

1. OpenCV, [equalizeHist](https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html).
2. OpenCV, [CLAHE class](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html).
3. OpenCV, [Mat::convertTo](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html) and [official OpenCV 5.0.0 release](https://github.com/opencv/opencv/releases/tag/5.0.0).
4. FLIR, [Tau 2 Product Specification, Rev. 141](https://flir.custhelp.com/ci/fattach/get/107854/0/filename/102-PS242-40-Tau2ProductSpec_Rev141.pdf), section 3.3.2.6.
5. Teledyne FLIR, [ADAS dataset formats](https://oem.flir.com/solutions/automotive/adas-dataset-form/).
6. Wang et al., [Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), IEEE TIP 13(4), 600-612, 2004, DOI 10.1109/TIP.2003.819861. SSIM uses the prior full-frame reflected-border 11x11 Gaussian convention (sigma 1.5); MAE/RMSE use gray-level units and PSNR is averaged per image.

See README.md for setup, compilation, testing, complete reproduction and fresh-metric commands. Package pins, official SDK digest, compiler/build information, source/binary hashes, manifests and raw timings accompany the results. References checked 4 November 2026.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')

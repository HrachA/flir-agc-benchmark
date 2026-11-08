"""CUDA efficiency and numerical-fidelity report, generated from measured CSVs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .support import read_csv,write_csv

ROOT=Path(__file__).resolve().parents[1]
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def create_report(out,cfg):
    q=read_csv(out/'quality_summary.csv');t=read_csv(out/'latency_summary.csv');b=read_csv(out/'batch_summary.csv');parity=read_csv(out/'parity_summary.csv')
    lookup={(r['method'],r['split'],r['mode']):r for r in t};b={r['method']:r for r in b}
    qmap={(r['engine'],r['split'],r['method']):r for r in q}
    cpu=read_csv(ROOT/'reference_results/native_cpu_latency.csv')
    cpu={s['name']:min((r for r in cpu if r['method']==s['name'] and r['split']=='pooled' and r['engine'] in ['cpp_exact_1','cpp_exact_8']),key=lambda r:float(r['mean_ms'])) for s in cfg['methods']}
    env=json.loads((out/'environment_latency.json').read_text());audit=json.loads((out/'audit.json').read_text())
    val=[r for r in q if r['engine']=='python' and r['split']=='val'];chosen=[]
    for family in dict.fromkeys(s['family'] for s in cfg['methods']):chosen.append(max((r for r in val if r['family']==family),key=lambda r:float(r['ssim_mean'])))
    chosen.sort(key=lambda r:float(r['ssim_mean']),reverse=True);winner=chosen[0]['method']
    selected=[];all_rows=[]
    for r in chosen:
        n=r['method'];selected.append([n,f'{float(cpu[n]["mean_ms"]):.3f}',
            *[f'{float(lookup[(n,"pooled",mode)]["mean_ms"]):.4f}' for mode in ['device_single_ms','device_amortized_ms','host_roundtrip_ms']],
            f'{float(b[n]["amortized_ms_per_image"]):.4f}',f'{float(qmap[("cuda","test",n)]["ssim_mean"]):.4f}'])
    for s in cfg['methods']:
        n=s['name'];all_rows.append([n,*[f'{float(lookup[(n,"pooled",mode)]["mean_ms"]):.4f}' for mode in
                                      ['device_single_ms','device_amortized_ms','gpu_roundtrip_ms','host_roundtrip_ms']],
                                      f'{float(b[n]["amortized_ms_per_image"]):.4f}'])
    changed=sum(int(r['different_pixels']) for r in parity);maximum=max(int(r['max_abs_difference']) for r in parity)
    different_methods=[r['method'] for r in parity if int(r['different_pixels'])]
    batch_diffs=sum(int(r['different_pixels']) for r in read_csv(out/'batch_integrity.csv'))
    test=qmap[('cuda','test',winner)];single=float(lookup[(winner,'pooled','device_single_ms')]['mean_ms'])
    amort=float(lookup[(winner,'pooled','device_amortized_ms')]['mean_ms']);wall=float(lookup[(winner,'pooled','host_roundtrip_ms')]['mean_ms'])
    batch=float(b[winner]['amortized_ms_per_image']);cp=float(cpu[winner]['mean_ms'])
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),sharey=True)
    y=np.arange(len(chosen))
    for i,(label,values,color) in enumerate([
        ('Prior C++ CPU',[float(cpu[r['method']]['mean_ms']) for r in chosen],'#2878B5'),
        ('GPU resident, single replay',[float(lookup[(r['method'],'pooled','device_single_ms')]['mean_ms']) for r in chosen],'#48A36F'),
        ('GPU host round trip',[float(lookup[(r['method'],'pooled','host_roundtrip_ms')]['mean_ms']) for r in chosen],'#D9791F')]):
        ax.barh(y+(i-1)*.23,values,height=.21,label=label,color=color)
    ax.set_yticks(y,[r['method'] for r in chosen]);ax.invert_yaxis();ax.set_xlabel('Mean ms/image');ax.set_title('(a) CPU and single-image CUDA paths')
    ax.legend(frameon=False,fontsize=8)
    for i,(label,values,color) in enumerate([
        ('Resident, 16-replay mean',[float(lookup[(r['method'],'pooled','device_amortized_ms')]['mean_ms']) for r in chosen],'#48A36F'),
        ('Host round trip',[float(lookup[(r['method'],'pooled','host_roundtrip_ms')]['mean_ms']) for r in chosen],'#D9791F'),
        ('Four streams, amortized',[float(b[r['method']]['amortized_ms_per_image']) for r in chosen],'#8057A4')]):
        bx.scatter(values,y+(i-1)*.18,label=label,color=color,marker=['o','s','^'][i],s=30)
    bx.set_xlabel('Mean or amortized ms/image');bx.set_title('(b) Residency and transfer costs');bx.legend(frameon=False,fontsize=8)
    ax.grid(axis='x',alpha=.2);bx.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    for ext in ['png','pdf']:fig.savefig(figures/f'cuda_efficiency.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    report=f'''# CUDA FLIR AGC benchmark

## Main result
The validation-selected method remains **{winner}**. Its mean GPU-resident single-replay interval is **{single:.4f} ms/image**; the average over 16 queued graph replays is **{amort:.4f} ms/replay**. The complete CPU-memory-to-CPU-memory wrapper takes **{wall:.4f} ms/image**, while a four-stream group takes **{batch:.4f} ms/image amortized**. The earlier optimized C++ CPU baseline for this setting is **{cp:.4f} ms/image**. GPU residency and transfers must therefore be reported separately; device-only timing does not establish the response time of a host-image application.

CUDA reaches test SSIM **{float(test['ssim_mean']):.4f}** and PSNR **{float(test['psnr_db_mean']):.2f} dB**. Across all 3295 images and 18 settings there are **{changed:,} differing pixels** versus the original Python renderer, maximum difference **{maximum}** levels. Four-stream outputs differ from single-stream outputs in **{batch_diffs} pixels** on all 128 timing frames for every setting. Full quality scores, parity details and raw timing samples are archived.

## Hardware and software
GPU: **{env['gpu']}**, compute capability {env['compute_capability']}, {env['sm_count']} SMs, {env['global_memory_bytes']/2**30:.2f} GiB device memory. CPU: **{env['cpu']}**, {env['physical_cores']} physical cores / {env['logical_processors']} logical processors, {env['ram_gib']:.1f} GiB RAM; OS {env['platform']}. CUDA compiler/toolkit **12.4.131**, runtime API {env['runtime_version']}, driver API {env['driver_api_version']}; NVIDIA driver details are in the NVIDIA-SMI captures. A driver advertising CUDA 13.3 does not mean Toolkit 13.3 compiled this code.

The installed host compiler is MSVC 19.43. CUDA 12.4 predates that compiler and was built with the explicit `--allow-unsupported-compiler` and `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` overrides. This combination passed the supplied tests and full-image validation, but is not represented as an officially supported NVIDIA toolchain combination. Prefer a supported toolkit/compiler combination for production. Source, build cache and flags are archived. Kernels target SM 8.9, use -O3, disable FMA contraction, retain precise division/sqrt, and do not enable fast math.

Python {env['python'].split()[0]}; packages {env['packages']}. OpenCV is used for CPU reference rendering and metrics; the rendering backend itself uses custom CUDA/CUB and does not require an OpenCV CUDA build. The GPU runs under the Windows WDDM display driver with normal desktop scheduling. Clocks, power limits and affinity are unchanged; other GPU/display activity can affect timing. These are results for the recorded GPU and implementations, not a universal fastest-possible guarantee.

## Data and method definitions
The audit retains **{audit['accepted_by_split']['val']} validation and {audit['accepted_by_split']['test']} test pairs**, all 640x512; **{audit['excluded']} 8-bit test TIFFs** are excluded. Eligible TIFFs have uint16 storage, decoded values exceeding 255 and no value above 16383. They are paired by exact stem with same-size vendor JPEGs. Inputs are hashed and never modified. No alignment, resizing or per-image fitting to the JPEG target is used.

All 18 original settings are retained: 14-bit HE, HE after an 8-bit stretch, five CLAHE clip/grid combinations, four plateau caps, three linear-histogram tail settings, fixed manual mapping and three auto-bright contrasts. Parameters are selected by validation mean SSIM; all test-setting results are descriptive. The renderers are stateless and do not reconstruct proprietary FLIR temporal filtering, ACE, SSO or DDE. FLIR documents AGC-processed JPEGs without identifying the dataset's exact processing configuration [4].

## Selected settings in each family
{table(['Method','Prior CPU ms','GPU single ms','GPU queued/replay ms','GPU host roundtrip ms','4-stream amortized ms','CUDA test SSIM'],selected)}

The CPU figures are **archived measurements from the preceding native C++ experiment**, choosing the faster of its reference-order 1/8-thread profiles for each method. They use the same CPU, dimensions and exact 128-frame subset, but were not re-measured in this CUDA session. Treat ratios as implementation/context comparisons, not a controlled same-session CPU/GPU contest. Reference CPU environment is included in reference_results/cpu_environment.json.

## CUDA implementation
The GPU keeps raw input, output, intermediate buffers, histogram counters, CUB scratch storage and graph instances allocated across frames. A pre-instantiated compute graph contains the full algorithm, without image-dependent caching. A separate graph contains pinned-memory H2D copies, the same computation, and D2H copies. CUB provides exact integer histogram/scan/reduction operations [1]. A 14-bit histogram/CDF supplies percentiles; plateau caps and CDF mapping are performed on device. Auto-bright uses a uint64 intensity sum. Manual and final mapping operations are fused into coalesced output kernels.

CLAHE uses one 256-thread block per tile, eight warp-private 256-bin histograms, parallel clipping/redistribution/CDF generation and a fused bilinear interpolation kernel. Its clipping, residual redistribution, reflect-101 padding and rounding conventions follow the OpenCV CPU definition; attribution and license are in THIRD_PARTY_NOTICES.txt [3]. Floating-point operation order is explicitly controlled where needed. No image-dependent statistics are copied to the CPU during rendering.

The device-input/output pointer and stream API permits integration into an already-GPU-resident pipeline. Output is one uint8 grayscale image; downstream YOLO channel replication, floating-point conversion and detector inference are outside this benchmark.

## Timing definitions
The exact original **64 validation + 64 test images** are loaded in RAM. Every setting has **640 timing samples**, with 20 warm-ups and five seeded, randomized repeats. Timings exclude file reading, decoding, metrics, context construction, graph capture and allocation of persistent buffers.

- **GPU single replay:** upload precedes timing. CUDA events bracket one compute-graph launch, followed by stream synchronization. It includes the graph's complete rendering and any stream/dispatch gaps between the markers, not just the sum of individual kernel durations.
- **GPU queued replay average:** CUDA events bracket {cfg['device_replays']} queued compute-graph replays on the same resident input; elapsed time is divided by {cfg['device_replays']}. Every replay recomputes all statistics and output. This is an amortized microbenchmark, not single-request latency or a batch of distinct images.
- **GPU round-trip event interval:** events bracket H2D + compute + D2H in one stream. It excludes CPU copies between NumPy arrays and pinned staging buffers, Python overhead and output allocation. This is measured in a separate instrumented invocation.
- **Host round trip:** uninstrumented production render call measured with perf_counter_ns; includes a newly allocated output, Python/ctypes overhead, pageable-to-pinned CPU copy, H2D, compute, D2H, synchronization and pinned-to-output copy. No CUDA timing events are added to this measured call.
- **Four-stream group:** four distinct frames use independent workspaces/graphs/streams. Host staging is performed, graphs are queued before collecting outputs, and the synchronized wall time includes output allocation and staging copies. Mean batch time / 4 is an amortized throughput measure, not an individual frame's response latency. There are 160 batch samples per method. All 128 timing frames are also checked for batch-output equivalence.

CUDA events are outside the captured graphs; graph capture's implicit dependency events are not mistaken for recorded timing events. Event intervals can include GPU idle/scheduling gaps and have finite resolution [2]. No CUDA tests or quality scoring run concurrently with the timing stages. WDDM/desktop scheduling still prevents strict isolation.

## All configurations
{table(['Method','GPU single','GPU queued mean','GPU transfer+compute','Host roundtrip','4-stream/image'],all_rows)}

All times above are milliseconds. latency_summary.csv additionally reports medians, p95, standard deviations and throughput. The queued-replay p95 is a percentile of replay-group averages; it is not a per-request tail latency. batch_summary.csv preserves full four-image batch latency and throughput.

## Quality and integrity
Every image is rendered by the original Python reference and by CUDA, then compared pixel-for-pixel. Hash-validated cached reference similarity scores are reused only for identical predictions. Changed predictions are rescored against the vendor JPEG. `--no-cache` recomputes all reference metrics. SSIM uses the prior full-frame reflected-border Gaussian convention (11x11, sigma 1.5); PSNR is averaged per image; MAE/RMSE use grayscale units. Pearson correlation and histogram Wasserstein distance are also reported [5]. Sequence-macro scores accompany image means because neighboring frames are dependent. High reconstruction similarity does not establish detection accuracy or temperature calibration.

## Work and memory estimates
For N=W*H, B=16384 and G CLAHE tiles, global histogram pipelines have O(N+B) work; manual mapping and auto-bright have O(N); CLAHE has O(N+B+G*256). GPU parallelism changes execution time, not these asymptotic counts. CUB can use multiple internal kernels, so a pipeline stage is not necessarily one kernel launch. complexity_cuda.csv and complexity_scaling.csv give approximate stage/work models at five resolutions; these are not measured instruction counts or FLOPs.

workspace.csv gives actual explicitly allocated device-buffer bytes for each method, plus 3N bytes of pinned host staging. Only buffers needed by the selected family are allocated. Four streams require four independent workspaces. CUDA context, graph/event objects, driver allocations, allocator overhead and unrelated display memory are excluded; this is not a total GPU-memory or peak-memory measurement. Graph capture and initial allocations are excluded from steady-state timing.

The measured host path demonstrates why transfer-inclusive performance must be checked for small images. If the application already keeps images on the GPU, the resident graph path avoids those transfers. If each frame starts and ends in CPU memory, compare the host-roundtrip or streamed measurements with CPU implementations before selecting a backend.

![CUDA residency and transfer costs](figures/cuda_efficiency.png)

Suggested caption: AGC implementation efficiency on the recorded RTX 4070 Ti SUPER and Intel Core i7-13700K. Left: archived native CPU timing, GPU-resident single-graph intervals, and transfer-inclusive host wall time. Right: queued GPU replay averages, host round trips and four-stream amortized cost. Settings are validation-selected. Queued replays and four-stream averages represent amortized throughput; they are not single-request response latency. CPU measurements come from the preceding experiment.

## References
1. NVIDIA, [CUB device primitives](https://nvidia.github.io/cccl/cub/), histogram, scan and reduction APIs (the build uses the version bundled with CUDA 12.4).
2. NVIDIA, [CUDA 12.4 event timing](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__EVENT.html) and [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), streams, graphs, pinned memory and transfer minimization.
3. OpenCV, [CLAHE implementation and license](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/clahe.cpp).
4. Teledyne FLIR, [ADAS dataset formats](https://oem.flir.com/solutions/automotive/adas-dataset-form/) and [Tau 2 Product Specification, Rev. 141](https://flir.custhelp.com/ci/fattach/get/107854/0/filename/102-PS242-40-Tau2ProductSpec_Rev141.pdf), section 3.3.2.6.
5. Wang et al., [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), IEEE TIP 13(4), 600-612, 2004, DOI 10.1109/TIP.2003.819861.

References checked 4 November 2026. README.md provides compilation, tests, complete reproduction and device-resident integration instructions. Source/DLL hashes, compiler settings, manifests, CPU/GPU metadata and raw measurements accompany this report.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    costs=[]
    stages={'he':4,'he8':8,'clahe':6,'plateau':7,'linear_hist':4,'manual':1,'auto_bright':2}
    for w,h in [(160,128),(320,256),(640,512),(1280,1024),(1920,1536)]:
        for s in cfg['methods']:
            f=s['family'];n=w*h;g=s.get('grid',0)**2
            costs.append({'method':s['name'],'width':w,'height':h,'pixels':n,'conceptual_pipeline_stages':stages[f],
                          'work_model':'O(N)' if f in ['manual','auto_bright'] else 'O(N+B+G*256)' if f=='clahe' else 'O(N+B)',
                          'histogram_bins':0 if f in ['manual','auto_bright'] else 16384,'clahe_tiles':g if f=='clahe' else 0,
                          'note':'Stages include CUB operations which can launch multiple kernels; modeled resolutions beyond 640x512 are not timed.'})
    write_csv(out/'complexity_scaling.csv',costs)
    write_csv(out/'complexity_cuda.csv',[r for r in costs if r['width']==640 and r['height']==512])

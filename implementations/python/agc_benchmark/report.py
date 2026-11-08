"""Generate a short factual report and publication-friendly plots from archived CSVs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .methods import Renderer

LABELS={'he':'HE (14-bit histogram)','he8':'HE (8-bit stretched)', 'clahe':'CLAHE',
        'plateau':'Plateau HE', 'linear_hist':'Linear histogram',
        'manual':'Fixed manual mapping', 'auto_bright':'Auto-bright'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def create_report(root,out,pairs,config):
    from .run import read_csv,write_csv,load
    import cv2
    env=json.loads((out/'environment.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    quality=read_csv(out/'quality_summary.csv');selected=read_csv(out/'selected_methods.csv')
    latency=read_csv(out/'latency_summary.csv')
    q={(r['split'],r['method']):r for r in quality}
    t={(r['split'],r['method']):r for r in latency}
    best=selection['global_validation_winner'];test=q[('test',best)]
    chosen=sorted([r for r in selected if r['split']=='val'],key=lambda r:float(r['ssim_mean']),reverse=True)
    summary=[]
    for r in chosen:
        n=r['method'];v=q[('test',n)];speed=t[('pooled',n)]
        summary.append([n,f'{float(r["ssim_mean"]):.4f}',f'{float(v["ssim_mean"]):.4f}',
                        f'{float(v["psnr_db_mean"]):.2f}',f'{float(v["mae_mean"]):.2f}',
                        f'{float(speed["mean_ms"]):.3f}',f'{float(speed["p95_ms"]):.3f}'])
    rows=[(r['method'],f'{float(r["ssim_mean"]):.4f}',f'{float(r["psnr_db_mean"]):.2f}',
           f'{float(q[("test",r["method"])]["ssim_mean"]):.4f}')
          for r in sorted([r for r in quality if r['split']=='val'],key=lambda r:float(r['ssim_mean']),reverse=True)]
    complexity=read_csv(out/'complexity.csv')
    work_rows=[]
    for r in complexity:
        if r['method'] in {v['method'] for v in chosen}:
            work_rows.append([r['method'],r['work_proxy_formula'],f"{int(r['work_proxy_at_resolution']):,}",r['asymptotic']])
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.2),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
    y=np.arange(len(chosen))
    vs=np.array([float(r['ssim_mean']) for r in chosen]);ts=np.array([float(q[('test',r['method'])]['ssim_mean']) for r in chosen])
    for i in y:ax.plot([vs[i],ts[i]],[i,i],color='0.75',lw=1)
    ax.scatter(vs,y,label='Validation',color='#2878B5',s=35,zorder=3)
    ax.scatter(ts,y,label='Test',color='#D9791F',marker='s',s=30,zorder=3)
    ax.set_yticks(y,[LABELS[r['family']] for r in chosen]);ax.invert_yaxis()
    ax.set_xlabel('Mean SSIM');ax.set_title('(a) Vendor-JPEG similarity');ax.legend(frameon=False,loc='best')
    means=[float(t[('pooled',r['method'])]['mean_ms']) for r in chosen]
    p95=[float(t[('pooled',r['method'])]['p95_ms']) for r in chosen]
    bx.barh(y,means,height=.5,color='#2878B5',alpha=.8,label='Mean')
    bx.scatter(p95,y,color='#D9791F',marker='|',s=100,label='95th percentile',zorder=3)
    bx.set_xlabel('CPU rendering latency (ms/image)');bx.set_title('(b) Rendering cost');bx.legend(frameon=False)
    ax.grid(axis='x',alpha=.2);bx.grid(axis='x',alpha=.2);ax.set_axisbelow(True);bx.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    for suffix in ['png','pdf']:fig.savefig(figures/f'quality_latency.{suffix}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    records=read_csv(out/'per_image_metrics.csv');exrows=[]
    cols=['Vendor JPEG','linear_hist','he','plateau','clahe', 'auto_bright']
    family_best={r['family']:r['method'] for r in chosen}
    specs={r['name']:r for r in config['methods']}
    pair_index={(p['split'],p['image_id']):p for p in pairs}
    fig,axes=plt.subplots(2,len(cols),figsize=(14,4.8))
    for row,split in enumerate(['val','test']):
        candidates=sorted([r for r in records if r['split']==split and r['method']==best],key=lambda r:float(r['ssim']))
        r=candidates[len(candidates)//2];pair=pair_index[(split,r['image_id'])]
        raw=load(root/pair['tiff_path'],cv2.IMREAD_UNCHANGED)
        for col,family in enumerate(cols):
            if col==0:im=load(root/pair['jpeg_path'],cv2.IMREAD_GRAYSCALE);title='Vendor JPEG'
            else:
                name=family_best[family];im=Renderer(specs[name])(raw);title=name
            axes[row,col].imshow(im,cmap='gray',vmin=0,vmax=255);axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(title,fontsize=8)
        axes[row,0].text(-.03,.5,split,transform=axes[row,0].transAxes,rotation=90,va='center',ha='right')
        exrows.append({'split':split,'image_id':r['image_id'],'selection':'median SSIM for globally validation-selected method',
                       'winner_method':best,'winner_ssim':r['ssim']})
    fig.subplots_adjust(wspace=.02,hspace=.04,left=.02,right=.995,top=.92,bottom=.01)
    for suffix in ['png','pdf']:fig.savefig(figures/f'representative_examples.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig);write_csv(out/'example_manifest.csv',exrows)
    count={s:sum(p['split']==s for p in pairs) for s in ['val','test']}
    sample_latency=t[('pooled',best)]
    pareto=[r['method'] for r in chosen if not any(
        float(other['ssim_mean'])>=float(r['ssim_mean']) and
        float(t[('pooled',other['method'])]['mean_ms'])<=float(t[('pooled',r['method'])]['mean_ms']) and
        (float(other['ssim_mean'])>float(r['ssim_mean']) or
         float(t[('pooled',other['method'])]['mean_ms'])<float(t[('pooled',r['method'])]['mean_ms']))
        for other in chosen if other['method']!=r['method'])]
    report=f'''# FLIR ADAS v2 paired-image AGC benchmark

## Main result
The validation-selected method is **{best}**. On the eligible test pairs it reaches mean SSIM **{float(test['ssim_mean']):.4f}**, mean per-image PSNR **{float(test['psnr_db_mean']):.2f} dB**, MAE **{float(test['mae_mean']):.2f} gray levels**, and RMSE **{float(test['rmse_mean']):.2f} gray levels**. Its CPU rendering latency averages **{float(sample_latency['mean_ms']):.3f} ms/image**, with a 95th percentile of **{float(sample_latency['p95_ms']):.3f} ms**. This identifies the closest tested rendering by validation SSIM, not the vendor's actual algorithm.

## Data and protocol
The run uses {count['val']} validation and {count['test']} test TIFF/JPEG pairs. Files are matched by exact stem within each original split; dimensions must agree and no alignment, resizing, intensity fitting to a target, or reference-assisted processing is allowed. Native eligible dimensions: {', '.join(audit['dimensions'])}. The audit checks all {audit['total_tiffs']} TIFFs and excludes {audit['excluded']} files: {audit['excluded_reasons']}. Eligibility requires uint16 storage, a decoded maximum above 255 and no value above 16383. This is an operational high-bit-depth criterion, not proof of sensor calibration or usable 14-bit precision. Results apply to this eligible subset, not all original test frames. File hashes and exact exclusions are archived.

All {len(config['methods'])} configurations are fixed before quality scoring. Parameters and method families are selected only using validation image-mean SSIM; test rankings for all settings are descriptive. Test images are not used to fit parameters. No learned model or detector is trained. Consecutive frames are dependent: standard deviations describe image variability, not confidence intervals. Equal-weight per-sequence means are provided as a sensitivity analysis in quality_summary.csv. Validation/test are never pooled for similarity scores.

## Selected setting in each family
{table(['Method (chosen on val)','Val SSIM','Test SSIM','Test PSNR (dB)','Test MAE','CPU mean (ms)','CPU p95 (ms)'],summary)}

SSIM uses 11x11 Gaussian windows, sigma 1.5, population local moments, C1=(0.01*255)^2 and C2=(0.03*255)^2, reflected borders, and an average over the full image [4]. This follows our prior study convention but is not a bit-for-bit reproduction of the paper's original MATLAB valid-window SSIM. PSNR, MAE, RMSE, Pearson correlation, and grayscale histogram Wasserstein-1 distance provide complementary measures. PSNR is averaged per image, not calculated from pooled MSE; no image is resized for scoring. High SSIM does not establish good detection accuracy or radiometric accuracy.

The selected plateau setting has test Pearson correlation {float(test['correlation_mean']):.4f} but a remaining MAE of {float(test['mae_mean']):.2f} levels: strong intensity co-variation therefore does not imply an exact vendor-image match. The table also shows that CLAHE improves SSIM over ordinary HE while having lower PSNR, so structural similarity and absolute pixel fidelity give different rankings. Plateau HE's remaining gap should not be attributed to any one unpublished camera-processing stage from these aggregate results alone.

Within the validation-selected family settings, the validation-SSIM versus CPU-mean-latency Pareto set is: {', '.join(pareto)}. A method is excluded from this set if another selected setting is at least as similar on validation and no slower, with one strict improvement. This is a descriptive trade-off, not a claim that the algorithms are optimized equally. Image-macro and sequence-macro scores should both be checked because long sequences can dominate frame averages. Eligible data contain {q[('val',best)]['sequences']} validation and {q[('test',best)]['sequences']} test sequences.

## Methods and interpretation
HE on the 14-bit histogram uses a cumulative mapping with the first occupied cumulative count subtracted. A second HE baseline applies OpenCV equalization after 0.1-99.9 percentile conversion to uint8. Five CLAHE configurations combine (clip limit, grid per axis) = (1,4), (2,8), (4,8), (2,16), and (4,16) on that same 8-bit stretch [3]. A grid of 8 means 8x8 tiles, not 8x8 pixels. The stretch cost is included.

The plateau implementation clips 16384-bin histogram counts at 10, 100, 240, or 1000 after 0.1% tail rejection, then maps the normalized clipped CDF to uint8. Clipped mass is discarded rather than redistributed. It is a simplified, stateless software implementation inspired by the documented plateau method [2]; it does not reproduce proprietary maximum-gain limiting, ACE, SSO, DDE, ROI tuning, temporal IIR or camera firmware quantization.

Three additional documented Tau 2 mode families are tested [2]: linear histogram mapping (0.1%, 1%, or 5% symmetric tails, midpoint 127); fixed manual mapping (contrast 1, brightness 8192); and auto-bright (contrast 8, 32, or 64). Manual uses y=clip(round[127+(I-brightness)*contrast/64]); auto-bright replaces brightness with the frame mean. Linear histogram scales the retained range to 255 gray levels around midpoint 127. Integer histogram percentiles use linear order-statistic interpolation. All custom output rounding uses nearest-even; optional firmware filtering and enhancements are omitted. The tested parameter grid is not claimed to be FLIR defaults or exhaustive.

FLIR identifies the dataset thermal formats as pre-AGC TIFF and AGC-processed JPEG [1]. Its public dataset page does not identify the exact AGC mode or settings. Missing temporal state, optional detail processing, JPEG compression, and the tested finite parameter grid can all limit agreement. In particular, a global monotonic map cannot reproduce arbitrary spatially varying contrast changes. These comparisons measure vendor-image reconstruction rather than recovering the camera's processing pipeline.

## CPU timing and environment
CPU: **{env['cpu']}**, {env['physical_cores']} physical cores / {env['logical_processors']} logical processors; RAM {env['ram_gib']:.1f} GiB; OS {env['platform']}. Registry nominal frequency: {env['registry_nominal_mhz']} MHz; actual boost frequency is not controlled. Python {platform_python(env)}; packages {env['packages']}. No GPU or OpenCL is used. OpenCV uses {env['opencv_threads']} thread(s). Quality scoring uses {config['quality_workers']} outer workers; latency is measured sequentially in one process, separately from metrics.

Each algorithm receives the same seeded subset of up to {config['latency_images_per_split']} eligible frames per split, loaded into RAM. After {config['latency_warmup']} warm-ups per configuration, {config['latency_repeats']} repetitions randomize frame and algorithm order. Timing encloses all rendering, histogram construction, stretching, output allocation and quantization, but excludes disk I/O, decoding, metrics, plot generation, and one-time renderer construction. Latency samples are in latency_samples.csv. Pooled latency mixes the two equally sampled split subsets; split-specific timings are also provided. Desktop background load and uncontrolled scheduling/boost mean these are host-specific measurements, not hardware-independent complexity.

The timings characterize these Python/NumPy/OpenCV implementations; they are not measurements of in-camera FLIR firmware or dedicated image-processing hardware.

## Approximate algorithmic work
Let N=width*height (327680 for 640x512), B=16384 histogram bins, and G=number of CLAHE tiles. Fixed manual mapping and auto-bright are O(N). Global HE, percentile-linear mapping, and plateau HE are O(N+B). CLAHE adds tile histograms, clipping/CDFs and interpolation: O(N+B+G*256) here, including the 14-bit-to-8-bit preparation. At fixed B and tile count all methods scale linearly with pixel count. CLAHE work per pixel is greater because several neighboring tile mappings are interpolated.

complexity.csv records an explicit approximate pixel/bin-visit proxy (roughly one to four full-image passes plus histogram work), including the numeric estimate at 640x512. These are coarse work estimates, not exact CPU instructions, FLOPs, cycle counts, or measured hardware counters. NumPy/OpenCV kernels perform multiple arithmetic and memory operations per visit. Additional storage is O(N+B) for global methods and O(N+B+G*256) for CLAHE.

{table(['Selected method','Approximate work proxy','Visits at 640x512','Asymptotic cost'],work_rows)}

An image-pass equivalent counts a conceptual histogram, point mapping, mean reduction, or image traversal, rather than every memory load or vectorized arithmetic subexpression. For example, the HE proxy counts one histogram pass and one LUT-application pass, plus approximate bin processing. This makes the resolution dependence explicit without claiming implementation-independent primitive instruction counts.

## All fixed configurations
{table(['Method','Val SSIM','Val PSNR (dB)','Test SSIM (descriptive)'],rows)}

## Figures
![Validation-selected settings: similarity and CPU latency](figures/quality_latency.png)

Suggested caption: Vendor-JPEG reconstruction and CPU rendering cost for one validation-selected setting per method family. Left: image-mean SSIM on validation and the eligible test subset. Right: mean CPU latency and 95th-percentile markers from the separate in-memory benchmark. Connectors are not uncertainty intervals. Settings are selected by validation SSIM; no test tuning is performed.

![Representative paired examples](figures/representative_examples.png)

The example in each split has median SSIM under the globally validation-selected method; filenames and selection are recorded in example_manifest.csv. All displays use the fixed 0-255 grayscale range. The examples are illustrative and do not replace aggregate metrics.

## References
[1] Teledyne FLIR, [FREE Teledyne FLIR Thermal Dataset for Algorithm Training](https://oem.flir.com/solutions/automotive/adas-dataset-form/), Dataset File Format.

[2] FLIR, [Tau 2 Product Specification, Rev. 141](https://flir.custhelp.com/ci/fattach/get/107854/0/filename/102-PS242-40-Tau2ProductSpec_Rev141.pdf), section 3.3.2.6, pp. 26-36. The PDF is manufacturer-hosted; some browsers may download it instead of displaying it.

[3] OpenCV, [Histogram Equalization and CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html). OpenCV version-specific behavior is pinned in requirements-lock.txt and the build dump.

[4] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), IEEE Transactions on Image Processing 13(4), 600-612 (2004), DOI 10.1109/TIP.2003.819861.

References checked 4 November 2026. Code, config, manifests, environment, raw timing samples, and per-image metrics accompany this report. See the repository README for exact commands.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')

def platform_python(env):
    return env['python'].split()[0]

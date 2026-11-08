"""Reproducible CUDA AGC latency, transfer and full-image parity experiment."""
import argparse,csv,gzip,json,subprocess,threading,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np
from .backend import CudaRenderer,CudaBatch,info,DLL
from .reference_methods import Renderer
from .metrics import Reference
from .support import audit,environment,load,read_csv,write_csv,dump,sha

ROOT=Path(__file__).resolve().parents[1]
METRICS=['ssim','psnr_db','mae','rmse','correlation','hist_w1_graylevels']
TIMES=['device_single_ms','device_amortized_ms','gpu_roundtrip_ms','host_roundtrip_ms']

def subset(pairs):
    expected=read_csv(ROOT/'reference_results/latency_manifest.csv')
    lookup={(r['split'],r['image_id']):r for r in pairs};selected=[]
    for r in expected:
        p=lookup.get((r['split'],r['image_id']))
        if not p or p['tiff_sha256']!=r['tiff_sha256']:raise ValueError('Timing subset differs from the recorded data')
        selected.append(p)
    return selected

def timing(root,pairs,cfg,out):
    selected=subset(pairs);frames=[load(root/r['tiff_path'],cv2.IMREAD_UNCHANGED) for r in selected]
    renderers=[CudaRenderer(s) for s in cfg['methods']]
    write_csv(out/'workspace.csv',[{'method':s['name'],'device_buffer_bytes':r.device_workspace_bytes,
                                    'pinned_host_buffer_bytes':640*512*3,'width':640,'height':512}
                                    for s,r in zip(cfg['methods'],renderers)])
    for r in renderers:
        r.upload(frames[0])
        for i in range(cfg['latency_warmup']):r.device_time();r(frames[i%len(frames)])
    rng=np.random.default_rng(cfg['seed']);rows=[];start=time.perf_counter()
    for repeat in range(cfg['latency_repeats']):
        for j in rng.permutation(len(frames)):
            for k in rng.permutation(len(renderers)):
                r=renderers[k];r.upload(frames[j])
                single=r.device_time();amortized=r.device_time_many(cfg['device_replays'])
                before=time.perf_counter_ns();pred=r(frames[j]);wall=(time.perf_counter_ns()-before)/1e6
                del pred
                pred,gpu=r.render_timed(frames[j]);del pred
                rows.append({'method':cfg['methods'][k]['name'],'family':cfg['methods'][k]['family'],
                             'split':selected[j]['split'],'image_id':selected[j]['image_id'],'repeat':repeat,
                             'device_single_ms':single,'device_amortized_ms':amortized,
                             'gpu_roundtrip_ms':gpu,'host_roundtrip_ms':wall})
        print(f'CUDA timing round {repeat+1}/{cfg["latency_repeats"]}: {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(out/'latency_samples.csv',rows);write_csv(out/'latency_manifest.csv',selected)
    summary=[]
    for s in cfg['methods']:
        for split in ['val','test','pooled']:
            samples=[r for r in rows if r['method']==s['name'] and (split=='pooled' or r['split']==split)]
            for mode in TIMES:
                v=np.array([r[mode] for r in samples]);summary.append({'method':s['name'],'family':s['family'],'split':split,
                    'mode':mode,'samples':len(v),'mean_ms':float(v.mean()),'median_ms':float(np.median(v)),
                    'p95_ms':float(np.percentile(v,95)),'std_ms':float(v.std()),'images_per_second':float(1000/v.mean())})
    write_csv(out/'latency_summary.csv',summary)
    for r in renderers:r.close()

def batch_timing(root,pairs,cfg,out):
    selected=subset(pairs);frames=[load(root/r['tiff_path'],cv2.IMREAD_UNCHANGED) for r in selected]
    count=cfg['batch_streams'];batches=[frames[i:i+count] for i in range(0,len(frames),count)]
    if any(len(b)!=count for b in batches):raise ValueError('Timing subset not divisible by batch size')
    rng=np.random.default_rng(cfg['seed']);samples=[];integrity=[]
    start=time.perf_counter()
    # All methods get the same batch groups; only method/group order is randomized.
    for k in rng.permutation(len(cfg['methods'])):
        s=cfg['methods'][k];renderer=CudaBatch(s,count)
        for i in range(cfg['latency_warmup']):renderer(batches[i%len(batches)])
        for repeat in range(cfg['latency_repeats']):
            for j in rng.permutation(len(batches)):
                before=time.perf_counter_ns();outputs=renderer(batches[j]);elapsed=(time.perf_counter_ns()-before)/1e6
                samples.append({'method':s['name'],'family':s['family'],'repeat':repeat,'batch_index':int(j),
                                'batch_size':count,'batch_latency_ms':elapsed,'amortized_ms_per_image':elapsed/count})
        # Check all timing frames without contaminating the timed region.
        single=CudaRenderer(s)
        for j,inputs in enumerate(batches):
            for offset,(frame,pred) in enumerate(zip(inputs,renderer(inputs))):
                ref=single(frame);d=np.abs(pred.astype(np.int16)-ref.astype(np.int16))
                integrity.append({'method':s['name'],'image_id':selected[j*count+offset]['image_id'],
                                  'different_pixels':int(np.count_nonzero(d)),'max_abs_difference':int(d.max())})
        renderer.close();single.close()
        print(f'Batch-{count} timing {s["name"]}: {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(out/'batch_samples.csv',samples);write_csv(out/'batch_integrity.csv',integrity)
    summary=[]
    for s in cfg['methods']:
        v=np.array([r['batch_latency_ms'] for r in samples if r['method']==s['name']])
        summary.append({'method':s['name'],'batch_size':count,'samples':len(v),'mean_batch_ms':float(v.mean()),
            'median_batch_ms':float(np.median(v)),'p95_batch_ms':float(np.percentile(v,95)),
            'amortized_ms_per_image':float(v.mean()/count),'images_per_second':float(count*1000/v.mean())})
    write_csv(out/'batch_summary.csv',summary)

def quality(root,pairs,cfg,out,no_cache=False):
    cache={}
    if not no_cache:
        p=json.loads((ROOT/'reference_results/provenance.json').read_text())
        assert sha(ROOT/'agc_cuda/reference_methods.py')==p['methods.py']
        assert sha(ROOT/'agc_cuda/metrics.py')==p['metrics.py']
        assert sha(ROOT/'configs/default.json')==p['config_sha256']
        if cv2.getBuildInformation()!=(ROOT/'reference_results/python_opencv_build.txt').read_text():raise ValueError('OpenCV build differs: use --no-cache')
        old={(r['split'],r['image_id']):r for r in read_csv(ROOT/'reference_results/manifest.csv')}
        for r in pairs:
            ref=old[(r['split'],r['image_id'])]
            if any(ref[k+'_sha256']!=r[k+'_sha256'] for k in ['tiff','jpeg']):raise ValueError('Changed inputs: use --no-cache')
        with gzip.open(ROOT/'reference_results/per_image_metrics.csv.gz','rt',newline='',encoding='utf-8') as f:
            cache={(r['split'],r['image_id'],r['method']):{k:float(r[k]) for k in METRICS} for r in csv.DictReader(f)}
    local=threading.local()
    def evaluate(pair):
        if not hasattr(local,'renderers'):
            local.renderers=[CudaRenderer(s) for s in cfg['methods']];local.refs=[Renderer(s) for s in cfg['methods']]
        raw=load(root/pair['tiff_path'],cv2.IMREAD_UNCHANGED);reference=None;metrics=[];parity=[]
        def score(image):
            nonlocal reference
            if reference is None:reference=Reference(load(root/pair['jpeg_path'],cv2.IMREAD_GRAYSCALE))
            return reference.evaluate(image)
        for s,gpu,py in zip(cfg['methods'],local.renderers,local.refs):
            base=py(raw);prediction=gpu(raw);delta=np.abs(prediction.astype(np.int16)-base.astype(np.int16));changed=int(np.count_nonzero(delta))
            prefix={'split':pair['split'],'image_id':pair['image_id'],'sequence_id':pair['sequence_id'],'method':s['name'],'family':s['family']}
            py_metrics=cache[(pair['split'],pair['image_id'],s['name'])] if cache else score(base)
            gpu_metrics=py_metrics if changed==0 else score(prediction)
            metrics.extend([{**prefix,'engine':'python',**py_metrics},{**prefix,'engine':'cuda',**gpu_metrics}])
            parity.append({**prefix,'pixels':raw.size,'different_pixels':changed,'max_abs_difference':int(delta.max()),'mean_abs_difference':float(delta.mean())})
        return metrics,parity
    start=time.perf_counter()
    with (out/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as qf,(out/'parity_per_image.csv').open('w',newline='',encoding='utf-8') as pf,ThreadPoolExecutor(max_workers=cfg['quality_workers']) as pool:
        qw=pw=None
        for i,(metrics,parity) in enumerate(pool.map(evaluate,pairs),1):
            if qw is None:
                qw=csv.DictWriter(qf,fieldnames=list(metrics[0]));qw.writeheader();pw=csv.DictWriter(pf,fieldnames=list(parity[0]));pw.writeheader()
            qw.writerows(metrics);pw.writerows(parity)
            if i%200==0 or i==len(pairs):
                qf.flush();pf.flush();elapsed=time.perf_counter()-start
                print(f'CUDA parity {i}/{len(pairs)}: {elapsed:.1f}s, ETA {elapsed/i*(len(pairs)-i):.0f}s',flush=True)
    dump(out/'quality_protocol.json',{'cache_used':bool(cache),'every_image_method_compared':True,'changed_outputs_scored_fresh':True})

def summarize(out,cfg):
    records=read_csv(out/'per_image_metrics.csv');pairs=read_csv(out/'manifest.csv')
    expected={(r['split'],r['image_id'],s['name'],e) for r in pairs for s in cfg['methods'] for e in ['python','cuda']}
    actual=[(r['split'],r['image_id'],r['method'],r['engine']) for r in records]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Incomplete or duplicated quality records')
    groups=defaultdict(list)
    for r in records:groups[(r['engine'],r['split'],r['method'])].append(r)
    summary=[]
    for (engine,split,method),rows in sorted(groups.items()):
        row={'engine':engine,'split':split,'method':method,'family':rows[0]['family'],'images':len(rows)}
        for key in METRICS:
            v=np.array([float(r[key]) for r in rows]);seq=defaultdict(list)
            for r in rows:seq[r['sequence_id']].append(float(r[key]))
            row[key+'_mean']=float(v.mean());row[key+'_std']=float(v.std());row[key+'_sequence_macro']=float(np.mean([np.mean(v) for v in seq.values()]))
        summary.append(row)
    write_csv(out/'quality_summary.csv',summary)
    groups=defaultdict(list)
    for r in read_csv(out/'parity_per_image.csv'):groups[(r['split'],r['method'])].append(r)
    parity=[]
    for (split,method),rows in sorted(groups.items()):
        pixels=sum(int(r['pixels']) for r in rows);diff=sum(int(r['different_pixels']) for r in rows)
        parity.append({'split':split,'method':method,'images':len(rows),'pixels':pixels,'different_pixels':diff,
                       'different_pixel_percent':100*diff/pixels,'max_abs_difference':max(int(r['max_abs_difference']) for r in rows)})
    write_csv(out/'parity_summary.csv',parity)
    selection=[]
    for e in ['python','cuda']:
        best=max((r for r in summary if r['engine']==e and r['split']=='val'),key=lambda r:r['ssim_mean'])
        test=next(r for r in summary if r['engine']==e and r['split']=='test' and r['method']==best['method'])
        selection.append({'engine':e,'method':best['method'],'val_ssim':best['ssim_mean'],'test_ssim':test['ssim_mean'],'test_psnr_db':test['psnr_db_mean']})
    write_csv(out/'selection.csv',selection)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/full'));p.add_argument('--stage',choices=['all','audit','latency','batch','quality','report'],default='all')
    p.add_argument('--no-cache',action='store_true');a=p.parse_args();root=a.dataset_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/default.json').read_text());cfg.update(quality_workers=2,device_replays=16,batch_streams=4)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);dump(out/'config.json',cfg)
    env=environment(cfg);env.update(info());env.update(gpu_used=True,cuda_graphs=True,fast_math=False,
        compiler_compatibility_override='--allow-unsupported-compiler' in (ROOT/'build/CMakeCache.txt').read_text())
    dump(out/f'environment_{a.stage}.json',env)
    (out/'cmake_build_cache.txt').write_text((ROOT/'build/CMakeCache.txt').read_text(),encoding='utf-8')
    try:(out/f'nvidia_smi_{a.stage}.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True),encoding='utf-8')
    except Exception:pass
    if a.stage in ['all','audit']:pairs=audit(root,out)
    else:pairs=read_csv(out/'manifest.csv')
    sources=sorted((ROOT/'agc_cuda').glob('*.py'))+sorted((ROOT/'cuda').glob('*'))+[ROOT/'CMakeLists.txt',ROOT/'configs/default.json']
    dump(out/'provenance.json',{'manifest_sha256':sha(out/'manifest.csv'),'cuda_library_sha256':sha(DLL),
         'sources':{f.relative_to(ROOT).as_posix():sha(f) for f in sources if f.is_file()},'dataset_root':str(root)})
    if a.stage in ['all','latency']:timing(root,pairs,cfg,out)
    if a.stage in ['all','batch']:batch_timing(root,pairs,cfg,out)
    if a.stage in ['all','quality']:quality(root,pairs,cfg,out,a.no_cache)
    if a.stage in ['all','quality','report']:
        summarize(out,cfg)
        from .report import create_report
        create_report(out,cfg)
    print('Completed',a.stage,out,flush=True)

if __name__=='__main__':main()

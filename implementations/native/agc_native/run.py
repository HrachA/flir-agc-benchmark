"""Same-data native/Python output checks and CPU timing comparison."""
import argparse,csv,gzip,hashlib,json,platform,sys,threading,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from . import native
from .native import NativeRenderer
from .reference_methods import Renderer,cost
from .metrics import Reference
from .support import audit,dump,environment,load,read_csv,write_csv,sha

ROOT=Path(__file__).resolve().parents[1]
ENGINES={'python':(False,1),'cpp_exact_1':(False,1),'cpp_exact_8':(False,8),'cpp_fast_1':(True,1),'cpp_fast_8':(True,8)}
METRICS=['ssim','psnr_db','mae','rmse','correlation','hist_w1_graylevels']

def latency(root,pairs,cfg,out):
    ref=read_csv(ROOT/'reference_results/latency_manifest.csv')
    available={(p['split'],p['image_id']):p for p in pairs}
    selected=[]
    for p in ref:
        current=available.get((p['split'],p['image_id']))
        if current is None or current['tiff_sha256']!=p['tiff_sha256']:
            raise ValueError('Timing subset differs from the original benchmark; use the documented FLIR dataset')
        selected.append(current)
    frames=[load(root/p['tiff_path'],cv2.IMREAD_UNCHANGED) for p in selected]
    write_csv(out/'latency_manifest.csv',selected)
    renderers={name:[Renderer(s) if name=='python' else NativeRenderer(s,fast=fast) for s in cfg['methods']]
               for name,(fast,threads) in ENGINES.items()}
    rng=np.random.default_rng(cfg['seed']);samples=[];start=time.perf_counter()
    for repeat in range(cfg['latency_repeats']):
        for engine in rng.permutation(list(ENGINES)):
            native.set_threads(ENGINES[engine][1])
            # Warm each engine after a thread-pool change; configuration outside timing.
            for r in renderers[engine]:
                for k in range(cfg['latency_warmup']):r(frames[k%len(frames)])
            for j in rng.permutation(len(frames)):
                for k in rng.permutation(len(cfg['methods'])):
                    before=time.perf_counter_ns();prediction=renderers[engine][k](frames[j]);elapsed=(time.perf_counter_ns()-before)/1e6
                    del prediction
                    samples.append({'engine':engine,'method':cfg['methods'][k]['name'],'family':cfg['methods'][k]['family'],
                                    'split':selected[j]['split'],'image_id':selected[j]['image_id'],'repeat':repeat,'latency_ms':elapsed})
        print(f'Timing round {repeat+1}/{cfg["latency_repeats"]}: {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(out/'latency_samples.csv',samples)
    summary=[]
    for engine in ENGINES:
        for s in cfg['methods']:
            for split in ['val','test','pooled']:
                values=np.array([r['latency_ms'] for r in samples if r['engine']==engine and r['method']==s['name'] and (split=='pooled' or r['split']==split)])
                summary.append({'engine':engine,'method':s['name'],'family':s['family'],'split':split,'samples':len(values),
                                'mean_ms':float(values.mean()),'median_ms':float(np.median(values)),
                                'p95_ms':float(np.percentile(values,95)),'std_ms':float(values.std()),
                                'images_per_second':float(1000/values.mean()),'ns_per_pixel':float(values.mean()*1e6/(640*512))})
    base={(r['split'],r['method']):r['mean_ms'] for r in summary if r['engine']=='python'}
    for r in summary:r['speedup_vs_python']=base[(r['split'],r['method'])]/r['mean_ms']
    write_csv(out/'latency_summary.csv',summary)
    # Multi-thread determinism checked on all 128 timing images, not inferred.
    checks=[]
    for mode in ['exact','fast']:
        for s in cfg['methods']:
            r=NativeRenderer(s,fast=mode=='fast');dif=0
            for frame in frames:
                native.set_threads(1);a=r(frame)
                native.set_threads(8);b=r(frame)
                dif+=int(np.count_nonzero(a!=b))
            checks.append({'mode':mode,'method':s['name'],'images':len(frames),'different_pixels_threads_1_vs_8':dif})
    write_csv(out/'thread_determinism.csv',checks);native.set_threads(1)

def quality(root,pairs,cfg,out,no_cache=False):
    cache={}
    if not no_cache:
        prov=json.loads((ROOT/'reference_results/provenance.json').read_text())
        assert sha(ROOT/'agc_native/reference_methods.py')==prov['methods.py']
        assert sha(ROOT/'agc_native/metrics.py')==prov['metrics.py']
        assert sha(ROOT/'configs/default.json')==prov['config_sha256']
        if cv2.getBuildInformation()!=(ROOT/'reference_results/python_opencv_build.txt').read_text():
            raise ValueError('Python OpenCV build differs from cached metrics; use --no-cache')
        ref={(r['split'],r['image_id']):r for r in read_csv(ROOT/'reference_results/manifest.csv')}
        for p in pairs:
            old=ref.get((p['split'],p['image_id']))
            if not old or old['tiff_sha256']!=p['tiff_sha256'] or old['jpeg_sha256']!=p['jpeg_sha256']:
                raise ValueError('Reference cache input hashes differ; rerun with --no-cache')
        with gzip.open(ROOT/'reference_results/per_image_metrics.csv.gz','rt',newline='',encoding='utf-8') as f:
            cache={(r['split'],r['image_id'],r['method']):{key:float(r[key]) for key in METRICS} for r in csv.DictReader(f)}
    native.set_threads(1);local=threading.local()
    def evaluate(pair):
        if not hasattr(local,'baseline'):
            local.baseline=[Renderer(s) for s in cfg['methods']]
            local.exact=[NativeRenderer(s,fast=False) for s in cfg['methods']]
            local.fast=[NativeRenderer(s,fast=True) for s in cfg['methods']]
        raw=load(root/pair['tiff_path'],cv2.IMREAD_UNCHANGED)
        reference=None
        def score(pred):
            nonlocal reference
            if reference is None:reference=Reference(load(root/pair['jpeg_path'],cv2.IMREAD_GRAYSCALE))
            return reference.evaluate(pred)
        rows=[];parity=[]
        for s,py,exact,fast in zip(cfg['methods'],local.baseline,local.exact,local.fast):
            before=py(raw);identity=(pair['split'],pair['image_id'],s['name'])
            base_score=cache[identity] if cache else score(before)
            prefix={'split':pair['split'],'image_id':pair['image_id'],'sequence_id':pair['sequence_id'],
                    'method':s['name'],'family':s['family']}
            rows.append({**prefix,'engine':'python',**base_score})
            for name,renderer in [('cpp_exact_1',exact),('cpp_fast_1',fast)]:
                prediction=renderer(raw)
                delta=np.abs(prediction.astype(np.int16)-before.astype(np.int16))
                changed=int(np.count_nonzero(delta));maximum=int(delta.max())
                parity.append({**prefix,'engine':name,'pixels':raw.size,'different_pixels':changed,
                               'max_abs_difference':maximum,'mean_abs_difference':float(delta.mean())})
                # Exact equality proves that every deterministic reference metric is
                # unchanged. Changed images are scored afresh against the vendor JPEG.
                metrics=base_score if changed==0 else score(prediction)
                rows.append({**prefix,'engine':name,**metrics})
        return rows,parity
    start=time.perf_counter();quality_path=out/'per_image_metrics.csv';parity_path=out/'parity_per_image.csv'
    with quality_path.open('w',newline='',encoding='utf-8') as qfile,parity_path.open('w',newline='',encoding='utf-8') as pfile,ThreadPoolExecutor(max_workers=cfg['quality_workers']) as pool:
        qw=pw=None
        for i,(rows,parity) in enumerate(pool.map(evaluate,pairs),1):
            if qw is None:
                qw=csv.DictWriter(qfile,fieldnames=list(rows[0]));qw.writeheader()
                pw=csv.DictWriter(pfile,fieldnames=list(parity[0]));pw.writeheader()
            qw.writerows(rows);pw.writerows(parity)
            if i%200==0 or i==len(pairs):
                qfile.flush();pfile.flush();elapsed=time.perf_counter()-start
                print(f'Parity/quality {i}/{len(pairs)}: {elapsed:.1f}s, ETA {elapsed/max(i,1)*(len(pairs)-i):.0f}s',flush=True)
    dump(out/'quality_protocol.json',{'reference_metric_cache_used':bool(cache),'full_dataset_pixel_comparison':True,
         'exact_matches_reuse_metrics':True,'changed_predictions_scored_against_vendor_jpeg':True,
         'native_quality_threads':1,'multi_thread_check':'all 128 timing images; thread_determinism.csv'})

def summarize(out,cfg):
    metrics=read_csv(out/'per_image_metrics.csv');manifest=read_csv(out/'manifest.csv');parity=read_csv(out/'parity_per_image.csv')
    expected={(p['split'],p['image_id'],s['name'],e) for p in manifest for s in cfg['methods'] for e in ['python','cpp_exact_1','cpp_fast_1']}
    actual=[(r['split'],r['image_id'],r['method'],r['engine']) for r in metrics]
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Incomplete or duplicate quality results')
    groups=defaultdict(list)
    for r in metrics:groups[(r['engine'],r['split'],r['method'])].append(r)
    rows=[]
    for (engine,split,method),items in sorted(groups.items()):
        row={'engine':engine,'split':split,'method':method,'family':items[0]['family'],'images':len(items)}
        for key in METRICS:
            values=np.array([float(r[key]) for r in items]);seq=defaultdict(list)
            for r in items:seq[r['sequence_id']].append(float(r[key]))
            row[key+'_mean']=float(values.mean());row[key+'_std']=float(values.std())
            row[key+'_sequence_macro']=float(np.mean([np.mean(v) for v in seq.values()]))
        rows.append(row)
    write_csv(out/'quality_summary.csv',rows)
    groups=defaultdict(list)
    for r in parity:groups[(r['engine'],r['split'],r['method'])].append(r)
    summary=[]
    for (engine,split,method),items in sorted(groups.items()):
        pixels=sum(int(r['pixels']) for r in items);changed=sum(int(r['different_pixels']) for r in items)
        summary.append({'engine':engine,'split':split,'method':method,'images':len(items),'pixels':pixels,
                        'images_with_differences':sum(int(r['different_pixels'])>0 for r in items),
                        'different_pixels':changed,'different_pixel_percent':100*changed/pixels,
                        'max_abs_difference':max(int(r['max_abs_difference']) for r in items),
                        'mean_abs_difference':sum(float(r['mean_abs_difference'])*int(r['pixels']) for r in items)/pixels})
    write_csv(out/'parity_summary.csv',summary)
    selected=[]
    for engine in ['python','cpp_exact_1','cpp_fast_1']:
        val=[r for r in rows if r['engine']==engine and r['split']=='val']
        winner=max(val,key=lambda r:r['ssim_mean'])
        selected.append({'engine':engine,'selection':'highest validation image-mean SSIM',
                         'method':winner['method'],'val_ssim':winner['ssim_mean'],
                         'test_ssim':next(r['ssim_mean'] for r in rows if r['engine']==engine and r['method']==winner['method'] and r['split']=='test')})
    write_csv(out/'selection.csv',selected)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/full'))
    p.add_argument('--stage',choices=['all','audit','latency','quality','report'],default='all')
    p.add_argument('--no-cache',action='store_true',help='Recompute every reference similarity metric')
    a=p.parse_args();root=a.dataset_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/default.json').read_text());cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);native.set_threads(1)
    dump(out/'config.json',cfg)
    env=environment(cfg);env.update(native.info());env.pop('opencv_build',None)
    dump(out/f'environment_{a.stage}.json',env)
    (out/'native_opencv_build.txt').write_text(native.info()['opencv_build'],encoding='utf-8')
    (out/'python_opencv_build.txt').write_text(cv2.getBuildInformation(),encoding='utf-8')
    if (ROOT/'build/CMakeCache.txt').exists():
        (out/'cmake_build_cache.txt').write_text((ROOT/'build/CMakeCache.txt').read_text(),encoding='utf-8')
    if a.stage in ['all','audit']:pairs=audit(root,out)
    else:pairs=read_csv(out/'manifest.csv')
    source_files=sorted((ROOT/'agc_native').glob('*.py'))+sorted(p for p in (ROOT/'cpp').iterdir() if p.is_file())+[ROOT/'CMakeLists.txt',ROOT/'configs/default.json']
    dump(out/'provenance.json',{'dataset_root':str(root),'manifest_sha256':sha(out/'manifest.csv'),
         'dll_sha256':sha(Path(native.info()['library'])),'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in source_files},
         'sdk_download':json.loads((ROOT/'third_party/opencv_download.json').read_text()) if (ROOT/'third_party/opencv_download.json').exists() else None})
    write_csv(out/'complexity.csv',[cost(s) for s in cfg['methods']])
    if a.stage in ['all','latency']:latency(root,pairs,cfg,out)
    if a.stage in ['all','quality']:quality(root,pairs,cfg,out,a.no_cache)
    if a.stage in ['all','quality','report']:
        summarize(out,cfg)
        from .report import create_report
        create_report(out,cfg)
    print('Completed',a.stage,out,flush=True)

if __name__=='__main__':main()

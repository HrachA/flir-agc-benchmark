"""python -m agc_benchmark.run --dataset-root PATH --output results/full"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import psutil
from PIL import Image
from .methods import Renderer, cost
from .metrics import Reference

SPLITS = {'val':'images_thermal_val','test':'video_thermal_test'}

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()

def load(path, flags):
    # np.fromfile supports Unicode Windows paths and preserves unchanged TIFF data.
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),flags)
    if image is None: raise ValueError(f'Could not decode {path}')
    return image

def environment(config):
    cpu=platform.processor()
    nominal=None
    if sys.platform=='win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
            nominal=winreg.QueryValueEx(key,'~MHz')[0]
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: revision=None
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'cpu':cpu,
            'physical_cores':psutil.cpu_count(logical=False), 'logical_processors':psutil.cpu_count(),
            'cpu_affinity':psutil.Process().cpu_affinity() if hasattr(psutil.Process(),'cpu_affinity') else None,
            'registry_nominal_mhz':nominal, 'ram_gib':round(psutil.virtual_memory().total/2**30,3),
            'platform':platform.platform(), 'python':sys.version, 'python_executable':sys.executable,
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','opencv-python-headless','Pillow','matplotlib','psutil']},
            'opencv_threads':cv2.getNumThreads(), 'opencv_opencl':cv2.ocl.useOpenCL(),
            'quality_workers':config['quality_workers'], 'git_revision':revision,
            'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
            'timing_clock':'time.perf_counter_ns', 'gpu_used':False}

def audit(root, out):
    accepted=[]; excluded=[]; inventory=[]
    for split, folder in SPLITS.items():
        tiffs=sorted((root/folder/'analyticsData').rglob('*'))
        tiffs=[p for p in tiffs if p.suffix.lower() in {'.tif','.tiff'}]
        if not tiffs: raise ValueError(f'No TIFFs for {split}: {root/folder}')
        index=defaultdict(list)
        for p in (root/folder/'data').rglob('*'):
            if p.suffix.lower() in {'.jpg','.jpeg'}: index[p.stem].append(p)
        for p in tiffs:
            row={'split':split, 'image_id':p.stem, 'tiff_path':p.relative_to(root).as_posix(),
                 'jpeg_path':'', 'width':'', 'height':'', 'tiff_dtype':'', 'bits_per_sample':'',
                 'raw_min':'','raw_max':'', 'status':'', 'reason':''}
            try:
                matches=index[p.stem]
                if len(matches)!=1: raise ValueError(f'JPEG match count {len(matches)}')
                jpg=matches[0]; row['jpeg_path']=jpg.relative_to(root).as_posix()
                raw=load(p,cv2.IMREAD_UNCHANGED)
                ref=load(jpg,cv2.IMREAD_GRAYSCALE)
                with Image.open(p) as im: bits=im.tag_v2.get(258)
                row.update(width=raw.shape[1],height=raw.shape[0],tiff_dtype=str(raw.dtype),
                           bits_per_sample=str(bits),raw_min=int(raw.min()),raw_max=int(raw.max()))
                if raw.ndim!=2: raise ValueError('TIFF is not single-channel')
                if raw.shape!=ref.shape: raise ValueError('TIFF/JPEG size mismatch; never resized')
                if raw.dtype!=np.uint16: raise ValueError('not_uint16_source')
                if raw.max()<=255: raise ValueError('uint16_but_only_8bit_range')
                if raw.max()>16383: raise ValueError('outside_nominal_14bit_range')
                row['status']='accepted'
                match=re.match(r'video-(.+)-frame-',p.stem)
                accepted.append({**row,'sequence_id':match.group(1) if match else p.stem,
                                 'tiff_sha256':sha(p),'jpeg_sha256':sha(jpg)})
            except Exception as e:
                row.update(status='excluded',reason=str(e)); excluded.append(row.copy())
            inventory.append(row)
        print(f'Audit {split}: {sum(x["split"]==split for x in accepted)} eligible / {len(tiffs)} TIFFs',flush=True)
    write_csv(out/'manifest.csv',accepted)
    write_csv(out/'inventory.csv',inventory)
    write_csv(out/'excluded.csv',excluded)
    summary={'total_tiffs':len(inventory),'accepted':len(accepted),'excluded':len(excluded),
             'accepted_by_split':dict(Counter(r['split'] for r in accepted)),
             'excluded_reasons':dict(Counter(r['reason'] for r in excluded)),
             'dimensions':dict(Counter(f'{r["width"]}x{r["height"]}' for r in accepted)),
             'criterion':'same-stem same-size grayscale TIFF/JPEG, uint16 TIFF, max >255 and <=16383; no resizing or alignment optimization'}
    dump(out/'audit.json',summary)
    if not all(any(r['split']==s for r in accepted) for s in SPLITS):
        raise ValueError('Need eligible pairs in both val and test')
    return accepted

def benchmark_latency(root, pairs, config, out):
    rng=np.random.default_rng(config['seed'])
    selected=[]
    for split in SPLITS:
        group=[p for p in pairs if p['split']==split]
        ids=rng.choice(len(group),min(config['latency_images_per_split'],len(group)),replace=False)
        selected.extend(group[i] for i in sorted(ids))
    frames=[load(root/p['tiff_path'],cv2.IMREAD_UNCHANGED) for p in selected]
    renderers=[Renderer(s) for s in config['methods']]
    for renderer in renderers:
        for i in range(config['latency_warmup']): renderer(frames[i%len(frames)])
    samples=[]
    start=time.perf_counter()
    for repeat in range(config['latency_repeats']):
        for j in rng.permutation(len(frames)):
            for k in rng.permutation(len(renderers)):
                t=time.perf_counter_ns(); pred=renderers[k](frames[j]); elapsed=(time.perf_counter_ns()-t)/1e6
                samples.append({'method':config['methods'][k]['name'],'split':selected[j]['split'],
                                'image_id':selected[j]['image_id'],'repeat':repeat,'latency_ms':elapsed})
                del pred
        print(f'Latency round {repeat+1}/{config["latency_repeats"]}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(out/'latency_samples.csv',samples)
    write_csv(out/'latency_manifest.csv',selected)
    rows=[]
    for s in config['methods']:
        for split in ['val','test','pooled']:
            t=np.array([v['latency_ms'] for v in samples if v['method']==s['name'] and (split=='pooled' or v['split']==split)])
            rows.append({'method':s['name'],'family':s['family'],'split':split,'samples':len(t),
                         'mean_ms':float(t.mean()),'median_ms':float(np.median(t)),
                         'p95_ms':float(np.percentile(t,95)),'std_ms':float(t.std()),
                         'images_per_second':float(1000/t.mean())})
    write_csv(out/'latency_summary.csv',rows)

def quality(root, pairs, config, out, resume=False):
    path=out/'per_image_metrics.csv'
    completed=set()
    if path.exists() and resume:
        groups=Counter((r['split'],r['image_id']) for r in read_csv(path))
        if any(n!=len(config['methods']) for n in groups.values()):
            raise ValueError('Incomplete image block in existing CSV; rerun quality without --resume')
        completed=set(groups)
    todo=[p for p in pairs if (p['split'],p['image_id']) not in completed]
    local=threading.local()
    def evaluate(pair):
        if not hasattr(local,'renderers'):
            local.renderers=[Renderer(s) for s in config['methods']]
        raw=load(root/pair['tiff_path'],cv2.IMREAD_UNCHANGED)
        ref=Reference(load(root/pair['jpeg_path'],cv2.IMREAD_GRAYSCALE))
        return [{'split':pair['split'],'image_id':pair['image_id'],'sequence_id':pair['sequence_id'],
                 'method':s['name'],'family':s['family'],**ref.evaluate(renderer(raw))}
                for s,renderer in zip(config['methods'],local.renderers)]
    mode='a' if resume and path.exists() else 'w'
    start=time.perf_counter()
    with path.open(mode,newline='',encoding='utf-8') as f, ThreadPoolExecutor(max_workers=config['quality_workers']) as pool:
        writer=None
        for i,rows in enumerate(pool.map(evaluate,todo),1):
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                if mode=='w': writer.writeheader()
            writer.writerows(rows)
            if i%100==0 or i==len(todo):
                f.flush(); elapsed=time.perf_counter()-start
                print(f'Quality {i+len(completed)}/{len(pairs)} pairs; {elapsed:.1f}s elapsed; {elapsed/max(i,1)*(len(todo)-i):.0f}s ETA',flush=True)

def aggregate(out, config):
    records=read_csv(out/'per_image_metrics.csv')
    manifest=read_csv(out/'manifest.csv')
    expected={(r['split'],r['image_id'],s['name']) for r in manifest for s in config['methods']}
    actual=[(r['split'],r['image_id'],r['method']) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Metrics are incomplete, duplicated or inconsistent with manifest/config; report generation refused')
    keys=['ssim','psnr_db','mae','rmse','correlation','hist_w1_graylevels']
    groups=defaultdict(list)
    for r in records:
        groups[(r['split'],r['method'])].append(r)
    rows=[]
    for (split,method),items in sorted(groups.items()):
        row={'split':split,'method':method,'family':items[0]['family'],'images':len(items),
             'sequences':len(set(r['sequence_id'] for r in items))}
        for key in keys:
            vals=np.array([float(r[key]) for r in items])
            seq=defaultdict(list)
            for r in items:seq[r['sequence_id']].append(float(r[key]))
            row[key+'_mean']=float(vals.mean());row[key+'_std']=float(vals.std())
            row[key+'_sequence_macro']=float(np.mean([np.mean(v) for v in seq.values()]))
        rows.append(row)
    write_csv(out/'quality_summary.csv',rows)
    by_name={(r['split'],r['method']):r for r in rows}
    selected=[]
    for family in dict.fromkeys(s['family'] for s in config['methods']):
        names=[s['name'] for s in config['methods'] if s['family']==family]
        name=max(names,key=lambda n:(by_name[('val',n)]['ssim_mean'],n))
        selected.extend({**by_name[(split,name)],'selection':'highest validation image-mean SSIM within family'} for split in SPLITS)
    write_csv(out/'selected_methods.csv',selected)
    winner=max((r for r in rows if r['split']=='val'),key=lambda r:r['ssim_mean'])['method']
    dump(out/'selection.json',{'primary_metric':'image-mean SSIM on validation',
                             'global_validation_winner':winner,
                             'test_evaluation':by_name[('test',winner)],
                             'no_test_parameter_tuning':True})
    return rows,selected

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/full'))
    p.add_argument('--config',type=Path,default=Path('configs/default.json'))
    p.add_argument('--stage',choices=['all','audit','latency','quality','report'],default='all')
    p.add_argument('--max-per-split',type=int,default=0,help='0=all eligible; positive for a smoke run')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=args.dataset_root.resolve();out=args.output.resolve()
    config=json.loads(args.config.read_text())
    if len(set(s['name'] for s in config['methods']))!=len(config['methods']):raise ValueError('Duplicate method name')
    cv2.setNumThreads(config['opencv_threads']);cv2.ocl.setUseOpenCL(False)
    out.mkdir(parents=True,exist_ok=True)
    run_cfg={**config,'max_per_split':args.max_per_split}
    cfgpath=out/'config.json'
    if cfgpath.exists() and (args.resume or args.stage not in ['all','audit']):
        if json.loads(cfgpath.read_text())!=run_cfg:raise ValueError('Config differs from archived run')
    dump(cfgpath,run_cfg)
    env=environment(config)
    dump(out/f'environment_{args.stage}.json',env)
    if not (out/'environment.json').exists():dump(out/'environment.json',env)
    (out/'opencv_build.txt').write_text(cv2.getBuildInformation(),encoding='utf-8')
    if args.stage in ['all','audit']:
        pairs=audit(root,out)
        if args.max_per_split:
            rng=np.random.default_rng(config['seed']); chosen=[]
            for split in SPLITS:
                group=[r for r in pairs if r['split']==split]
                ids=sorted(rng.choice(len(group),min(args.max_per_split,len(group)),replace=False))
                chosen.extend(group[i] for i in ids)
            pairs=chosen;write_csv(out/'manifest.csv',pairs)
    else:pairs=read_csv(out/'manifest.csv')
    dump(out/'run_identity.json',{'manifest_sha256':sha(out/'manifest.csv'),'config_sha256':sha(cfgpath),
                                'dataset_root':str(root),'number_of_pairs':len(pairs)})
    source_root=Path(__file__).resolve().parents[1]
    dump(out/'source_hashes.json',{str(p.relative_to(source_root)):sha(p)
                                for p in sorted((source_root/'agc_benchmark').glob('*.py'))})
    write_csv(out/'complexity.csv',[cost(s) for s in config['methods']])
    write_csv(out/'complexity_scaling.csv',[cost(s,w,h) for w,h in
              [(160,128),(320,256),(640,512),(1280,1024),(1920,1536)] for s in config['methods']])
    if args.stage in ['all','latency']:benchmark_latency(root,pairs,config,out)
    if args.stage in ['all','quality']:quality(root,pairs,config,out,args.resume)
    if args.stage in ['all','quality','report']:
        aggregate(out,config)
        from .report import create_report
        create_report(root,out,pairs,config)
    print(f'Completed stage {args.stage}: {out}',flush=True)

if __name__=='__main__':main()

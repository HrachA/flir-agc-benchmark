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
from .reference_methods import Renderer, cost
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


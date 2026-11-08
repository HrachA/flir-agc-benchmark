"""Verify a completed native experiment and unchanged paired source files."""
import argparse
from collections import Counter
from .support import read_csv,sha
from .run import ENGINES
from pathlib import Path
import json

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path);p.add_argument('--verify-inputs',action='store_true')
    a=p.parse_args();out=a.results;config=json.loads((out/'config.json').read_text())
    manifest=read_csv(out/'manifest.csv');quality=read_csv(out/'per_image_metrics.csv')
    expected={(r['split'],r['image_id'],s['name'],e) for r in manifest for s in config['methods']
              for e in ['python','cpp_exact_1','cpp_fast_1']}
    observed=[(r['split'],r['image_id'],r['method'],r['engine']) for r in quality]
    assert set(observed)==expected and len(observed)==len(expected),'Incomplete/duplicate quality rows'
    parity=read_csv(out/'parity_per_image.csv')
    assert len(parity)==len(manifest)*len(config['methods'])*2
    assert all(int(r['different_pixels'])==0 for r in parity if r['engine']=='cpp_exact_1'),'Reference-order C++ is not bit exact'
    samples=read_csv(out/'latency_samples.csv');counts=Counter((r['engine'],r['method']) for r in samples)
    assert set(counts.values())=={640} and set(counts)=={(e,s['name']) for e in ENGINES for s in config['methods']},'Incomplete timing rows'
    assert all(float(r['latency_ms'])>0 for r in samples)
    assert all(int(r['different_pixels_threads_1_vs_8'])==0 for r in read_csv(out/'thread_determinism.csv'))
    if a.verify_inputs:
        if not a.dataset_root:p.error('--verify-inputs needs --dataset-root')
        for r in manifest:
            for key in ['tiff','jpeg']:
                assert sha(a.dataset_root/r[key+'_path'])==r[key+'_sha256'],'Source data changed'
    print(f'PASS: {len(manifest)} pairs, {len(quality)} quality rows, {len(parity)} parity rows, {len(samples)} timing samples.')
    print('PASS: reference-order C++ is pixel-identical on the full dataset; timing-subset thread checks match.')
    if a.verify_inputs:print('PASS: every TIFF/JPEG SHA-256 still matches.')

if __name__=='__main__':main()

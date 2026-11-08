"""Read-only verification of a completed CUDA benchmark archive."""
import argparse,json
from collections import Counter
from pathlib import Path
from .support import read_csv,sha
from .run import TIMES

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path);p.add_argument('--verify-inputs',action='store_true');a=p.parse_args();out=a.results
    cfg=json.loads((out/'config.json').read_text());manifest=read_csv(out/'manifest.csv')
    metrics=read_csv(out/'per_image_metrics.csv');parity=read_csv(out/'parity_per_image.csv')
    expected={(r['split'],r['image_id'],s['name'],e) for r in manifest for s in cfg['methods'] for e in ['python','cuda']}
    actual=[(r['split'],r['image_id'],r['method'],r['engine']) for r in metrics]
    assert set(actual)==expected and len(actual)==len(expected),'Incomplete/duplicate quality records'
    assert len(parity)==len(manifest)*len(cfg['methods'])
    assert all(int(r['different_pixels'])==0 for r in parity),'CUDA output differs from reference; inspect parity before claiming exactness'
    samples=read_csv(out/'latency_samples.csv');counts=Counter(r['method'] for r in samples)
    assert set(counts.values())=={640} and set(counts)=={s['name'] for s in cfg['methods']}
    assert all(float(r[k])>0 for r in samples for k in TIMES)
    batches=read_csv(out/'batch_samples.csv');counts=Counter(r['method'] for r in batches)
    assert set(counts.values())=={160} and len(counts)==len(cfg['methods'])
    assert all(float(r['batch_latency_ms'])>0 for r in batches)
    assert all(int(r['different_pixels'])==0 for r in read_csv(out/'batch_integrity.csv'))
    if a.verify_inputs:
        if not a.dataset_root:p.error('--verify-inputs needs --dataset-root')
        for r in manifest:
            for key in ['tiff','jpeg']:assert sha(a.dataset_root/r[key+'_path'])==r[key+'_sha256'],'Source input changed'
    print(f'PASS: {len(manifest)} pairs, {len(metrics)} quality rows, {len(parity)} exact parity checks, {len(samples)} timing records, {len(batches)} batch records.')
    if a.verify_inputs:print('PASS: every paired source-file SHA-256 matches.')

if __name__=='__main__':main()

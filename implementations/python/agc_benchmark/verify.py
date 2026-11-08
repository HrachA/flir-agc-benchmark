"""Read-only integrity checks for a finished benchmark archive."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

def rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--verify-inputs',action='store_true')
    a=p.parse_args();out=a.results
    config=json.loads((out/'config.json').read_text())
    manifest=rows(out/'manifest.csv');data=rows(out/'per_image_metrics.csv')
    expected={(r['split'],r['image_id'],s['name']) for r in manifest for s in config['methods']}
    actual=[(r['split'],r['image_id'],r['method']) for r in data]
    assert len(actual)==len(set(actual)) and set(actual)==expected,'Incomplete or duplicate scores'
    for r in data:
        assert -1.000001<=float(r['ssim'])<=1.000001
        assert -1.000001<=float(r['correlation'])<=1.000001
        assert 0<=float(r['mae'])<=255 and 0<=float(r['rmse'])<=255
        assert 0<=float(r['hist_w1_graylevels'])<=255.000001
        assert math.isfinite(float(r['ssim']))
    val=[r for r in rows(out/'quality_summary.csv') if r['split']=='val']
    best=max(val,key=lambda r:float(r['ssim_mean']))['method']
    assert best==json.loads((out/'selection.json').read_text())['global_validation_winner']
    samples=rows(out/'latency_samples.csv')
    expected_per_method=sum(min(config['latency_images_per_split'],sum(r['split']==split for r in manifest))
                            for split in ['val','test'])*config['latency_repeats']
    counts=Counter(r['method'] for r in samples)
    assert set(counts)=={s['name'] for s in config['methods']}
    assert set(counts.values())=={expected_per_method}
    assert all(float(r['latency_ms'])>0 for r in samples)
    if a.verify_inputs:
        if a.dataset_root is None:p.error('--verify-inputs requires --dataset-root')
        for r in manifest:
            for key in ['tiff','jpeg']:
                assert digest(a.dataset_root/r[key+'_path'])==r[key+'_sha256'],f'Changed source: {r[key+"_path"]}'
    print(f'PASS: {len(manifest)} pairs, {len(data)} unique image/method scores, {len(samples)} positive timing samples; selection uses validation SSIM.')
    if a.verify_inputs:print('PASS: all paired source-file SHA-256 hashes match.')

if __name__=='__main__':main()

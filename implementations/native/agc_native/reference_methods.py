"""Explicit CPU renderers. FLIR-inspired methods are not camera firmware replicas."""
from __future__ import annotations
import cv2
import numpy as np

BINS = 16384

def histogram(raw):
    return np.bincount(raw.ravel(), minlength=BINS).astype(np.float64)

def quantile(hist, percent):
    """NumPy-linear percentile via exact integer histogram order statistics."""
    rank = (hist.sum() - 1) * percent / 100
    lo, hi = int(np.floor(rank)), int(np.ceil(rank))
    cdf = hist.cumsum()
    a, b = np.searchsorted(cdf, [lo + 1, hi + 1])
    return float(a + (b - a) * (rank - lo))

def u8(values):
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)

def stretch(raw, hist, tail):
    lo, hi = quantile(hist, tail), quantile(hist, 100-tail)
    if hi <= lo:
        return np.zeros(raw.shape, np.uint8)
    return u8((raw.astype(np.float32) - lo) * (255.0 / (hi-lo)))

def hist_lut(hist):
    cdf = hist.cumsum()
    nonzero = cdf[hist > 0]
    if len(nonzero) < 2:
        return np.zeros(BINS, np.uint8)
    return u8((cdf - nonzero[0]) * (255.0 / (cdf[-1]-nonzero[0])))

class Renderer:
    def __init__(self, spec):
        self.spec = spec
        self.clahe = None
        if spec['family'] == 'clahe':
            self.clahe = cv2.createCLAHE(spec['clip_limit'], (spec['grid'], spec['grid']))

    def __call__(self, raw):
        """Include histogram/normalization/all allocations; no image I/O or metrics."""
        s, f = self.spec, self.spec['family']
        if f == 'manual':
            return u8(127 + (raw.astype(np.float32)-s['brightness']) * s['contrast']/64)
        if f == 'auto_bright':
            return u8(127 + (raw.astype(np.float32)-raw.mean()) * s['contrast']/64)
        h = histogram(raw)
        if f == 'he':
            return hist_lut(h)[raw]
        if f == 'plateau':
            lo, hi = quantile(h, s['tail_percent']), quantile(h, 100-s['tail_percent'])
            # Retain intensity bins inside the percentile bounds. Clipped mass is
            # discarded, not redistributed: plateau HE, distinct from CLAHE.
            bins = np.arange(BINS)
            h[(bins < lo) | (bins > hi)] = 0
            h = np.minimum(h, s['plateau_count'])
            return hist_lut(h)[raw]
        if f == 'linear_hist':
            lo, hi = quantile(h, s['tail_percent']), quantile(h, 100-s['tail_percent'])
            if hi <= lo:
                return np.full(raw.shape, s['midpoint'], np.uint8)
            return u8(s['midpoint'] + (raw.astype(np.float32)-(lo+hi)/2) * 255/(hi-lo))
        base = stretch(raw, h, s['tail_percent'])
        if f == 'he8':
            return cv2.equalizeHist(base)
        if f == 'clahe':
            return self.clahe.apply(base)
        raise ValueError(f'Unknown method family: {f}')

def cost(spec, width=640, height=512):
    """Scalar-work proxies, not FLOPs/instructions; each pass touches N pixels."""
    n, b, f = width*height, BINS, spec['family']
    # Full image scans/evaluations are counted coarsely. Vectorized arithmetic
    # and histogram updates each have different actual machine costs.
    passes = {'he':2, 'he8':4, 'plateau':2, 'linear_hist':2,
              'manual':1, 'auto_bright':2, 'clahe':4}[f]
    bins = {'he':2*b, 'he8':3*b+512, 'plateau':5*b,
            'linear_hist':3*b, 'manual':0, 'auto_bright':0,
            'clahe':3*b+3*spec.get('grid',0)**2*256}[f]
    expr = f'{passes}N + {bins}'
    return {'method':spec['name'], 'family':f, 'width':width, 'height':height,
            'pixels':n, 'pixel_pass_equivalents':passes,
            'histogram_bin_work_proxy':bins,
            'work_proxy_at_resolution':passes*n+bins,
            'work_proxy_formula':expr,
            'asymptotic':'O(N + G*256 + 16384)' if f=='clahe' else ('O(N)' if not bins else 'O(N + 16384)'),
            'note':'Approximate pixel/bin visits; excludes implementation-dependent arithmetic, SIMD, cache and allocation costs.'}

"""Zero-copy input bridge to the C ABI; returns a newly allocated uint8 output."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'he':0,'he8':1,'clahe':2,'plateau':3,'linear_hist':4,'manual':5,'auto_bright':6}
_dll_dirs=[]
if os.name=='nt':
    sdk_roots=[ROOT/'third_party/opencv/build']
    cache=ROOT/'build/CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            if line.startswith('OpenCV_DIR:') and '=' in line:
                configured=Path(line.split('=',1)[1])
                sdk_roots.extend([configured,*list(configured.parents)[:3]])
    candidates=[]
    for sdk in sdk_roots:
        candidates.extend(sdk.glob('x64/*/bin'))
        if (sdk/'bin').is_dir():candidates.append(sdk/'bin')
    candidates=list(dict.fromkeys(candidates))
    for p in candidates:_dll_dirs.append(os.add_dll_directory(str(p)))
    path=ROOT/'build/Release/flir_agc.dll'
else:path=ROOT/'build/libflir_agc.so'
lib=ct.CDLL(str(path))
lib.agc_create.argtypes=[ct.c_int,ct.c_double,ct.c_double,ct.c_int,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_int]
lib.agc_create.restype=ct.c_void_p
lib.agc_render.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p];lib.agc_render.restype=ct.c_int
lib.agc_destroy.argtypes=[ct.c_void_p];lib.agc_destroy.restype=None
lib.agc_last_error.restype=ct.c_char_p
lib.agc_set_threads.argtypes=[ct.c_int];lib.agc_set_threads.restype=None
lib.agc_opencv_info.restype=ct.c_char_p;lib.agc_compiler.restype=ct.c_char_p

def set_threads(n):lib.agc_set_threads(n)
def info():return {'compiler':lib.agc_compiler().decode(),'opencv_build':lib.agc_opencv_info().decode(),'library':str(path)}

class NativeRenderer:
    def __init__(self,spec,fast=False):
        self.spec=spec
        self.handle=lib.agc_create(FAMILIES[spec['family']],spec.get('tail_percent',0),spec.get('clip_limit',2),
                    spec.get('grid',8),spec.get('plateau_count',240),spec.get('midpoint',127),
                    spec.get('contrast',1),spec.get('brightness',8192),int(fast))
        if not self.handle:raise RuntimeError(lib.agc_last_error().decode())
    def __call__(self,raw,out=None):
        if raw.dtype!=np.uint16 or raw.ndim!=2 or not raw.flags.c_contiguous:
            raise ValueError('Native input must be a contiguous 2D uint16 array')
        if out is None:out=np.empty(raw.shape,np.uint8)
        elif out.shape!=raw.shape or out.dtype!=np.uint8 or not out.flags.c_contiguous:
            raise ValueError('Output must be a contiguous matching uint8 buffer')
        result=lib.agc_render(self.handle,raw.ctypes.data,raw.shape[1],raw.shape[0],out.ctypes.data)
        if result:raise RuntimeError(lib.agc_last_error().decode())
        return out
    def close(self):
        if self.handle:lib.agc_destroy(self.handle);self.handle=None
    def __del__(self):
        if getattr(self,'handle',None):self.close()

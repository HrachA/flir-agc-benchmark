import ctypes as ct
import json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'he':0,'he8':1,'clahe':2,'plateau':3,'linear_hist':4,'manual':5,'auto_bright':6}
_dirs=[]
if os.name=='nt':
    cuda=Path(os.environ.get('CUDA_PATH',r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'))
    cache=ROOT/'build/CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            if line.startswith('CUDAToolkit_BIN_DIR:') and '=' in line:
                cuda=Path(line.split('=',1)[1]).parent
    if (cuda/'bin').exists():_dirs.append(os.add_dll_directory(str(cuda/'bin')))
    DLL=ROOT/'build/Release/flir_agc_cuda.dll'
else:DLL=ROOT/'build/libflir_agc_cuda.so'
lib=ct.CDLL(str(DLL))
lib.agc_cuda_create.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_int,ct.c_int,ct.c_int,ct.c_double,ct.c_double]
lib.agc_cuda_create.restype=ct.c_void_p
lib.agc_cuda_destroy.argtypes=[ct.c_void_p];lib.agc_cuda_destroy.restype=None
lib.agc_cuda_render.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.POINTER(ct.c_float)];lib.agc_cuda_render.restype=ct.c_int
lib.agc_cuda_upload.argtypes=[ct.c_void_p,ct.c_void_p];lib.agc_cuda_upload.restype=ct.c_int
lib.agc_cuda_device_time.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];lib.agc_cuda_device_time.restype=ct.c_int
lib.agc_cuda_device_time_many.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_float)];lib.agc_cuda_device_time_many.restype=ct.c_int
lib.agc_cuda_batch_render.argtypes=[ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p),ct.c_int];lib.agc_cuda_batch_render.restype=ct.c_int
lib.agc_cuda_workspace.argtypes=[ct.c_void_p];lib.agc_cuda_workspace.restype=ct.c_size_t
lib.agc_cuda_error.restype=ct.c_char_p;lib.agc_cuda_info.restype=ct.c_char_p

def check(rc):
    if rc:raise RuntimeError(lib.agc_cuda_error().decode())
def info():
    raw=lib.agc_cuda_info()
    if not raw:raise RuntimeError(lib.agc_cuda_error().decode())
    return json.loads(raw)

class CudaRenderer:
    def __init__(self,spec,width=640,height=512):
        self.shape=(height,width)
        self.handle=lib.agc_cuda_create(width,height,FAMILIES[spec['family']],spec.get('tail_percent',0.1),
            spec.get('clip_limit',2),spec.get('grid',8),spec.get('plateau_count',240),spec.get('midpoint',127),
            spec.get('contrast',1),spec.get('brightness',8192))
        if not self.handle:raise RuntimeError(lib.agc_cuda_error().decode())
    def validate(self,raw):
        if raw.dtype!=np.uint16 or raw.shape!=self.shape or not raw.flags.c_contiguous:
            raise ValueError('Expected matching contiguous uint16 image; range must be 0..16383')
    def render_timed(self,raw):
        self.validate(raw);out=np.empty(self.shape,np.uint8);gpu=ct.c_float()
        check(lib.agc_cuda_render(self.handle,raw.ctypes.data,out.ctypes.data,ct.byref(gpu)))
        return out,gpu.value
    def __call__(self,raw):
        self.validate(raw);out=np.empty(self.shape,np.uint8)
        check(lib.agc_cuda_render(self.handle,raw.ctypes.data,out.ctypes.data,None))
        return out
    def upload(self,raw):
        self.validate(raw);check(lib.agc_cuda_upload(self.handle,raw.ctypes.data))
    def device_time(self):
        ms=ct.c_float();check(lib.agc_cuda_device_time(self.handle,ct.byref(ms)));return ms.value
    def device_time_many(self,repeats=16):
        ms=ct.c_float();check(lib.agc_cuda_device_time_many(self.handle,repeats,ct.byref(ms)));return ms.value
    @property
    def device_workspace_bytes(self):return lib.agc_cuda_workspace(self.handle)
    def close(self):
        if self.handle:lib.agc_cuda_destroy(self.handle);self.handle=None
    def __del__(self):
        if getattr(self,'handle',None):self.close()

class CudaBatch:
    def __init__(self,spec,count=4,width=640,height=512):
        self.renderers=[CudaRenderer(spec,width,height) for _ in range(count)]
        self.handles=(ct.c_void_p*count)(*[r.handle for r in self.renderers]);self.count=count
    def __call__(self,frames):
        if len(frames)!=self.count:raise ValueError('Batch count mismatch')
        for r,frame in zip(self.renderers,frames):r.validate(frame)
        outputs=[np.empty(frame.shape,np.uint8) for frame in frames]
        inputs=(ct.c_void_p*self.count)(*[frame.ctypes.data for frame in frames])
        out_ptrs=(ct.c_void_p*self.count)(*[out.ctypes.data for out in outputs])
        check(lib.agc_cuda_batch_render(self.handles,inputs,out_ptrs,self.count))
        return outputs
    def close(self):
        for r in self.renderers:r.close()

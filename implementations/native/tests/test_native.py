import json,unittest
from pathlib import Path
import cv2
import numpy as np
from agc_native.native import NativeRenderer,set_threads
from agc_native.reference_methods import Renderer

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs=json.loads(Path('configs/default.json').read_text())['methods']
        cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);set_threads(1)

    def test_exact_full_range_and_constant(self):
        for raw in [np.arange(16384,dtype=np.uint16).reshape(128,128),np.full((64,64),8000,np.uint16)]:
            for s in self.specs:
                np.testing.assert_array_equal(NativeRenderer(s)(raw),Renderer(s)(raw),err_msg=s['name'])

    def test_exact_random_odd_shape_and_source_preservation(self):
        raw=np.random.default_rng(11).integers(1000,12000,(73,61),dtype=np.uint16);original=raw.copy()
        for s in self.specs:
            np.testing.assert_array_equal(NativeRenderer(s)(raw),Renderer(s)(raw),err_msg=s['name'])
            np.testing.assert_array_equal(raw,original)

    def test_fast_range_shape(self):
        raw=np.random.default_rng(5).integers(3000,12000,(512,640),dtype=np.uint16)
        for s in self.specs:
            pred=NativeRenderer(s,fast=True)(raw)
            self.assertEqual(pred.dtype,np.uint8);self.assertEqual(pred.shape,raw.shape)

    def test_invalid_input_rejected(self):
        render=NativeRenderer(self.specs[0])
        with self.assertRaises(ValueError):render(np.zeros((64,64),np.uint8))
        with self.assertRaises(RuntimeError):render(np.full((64,64),40000,np.uint16))

    def test_thread_invariance(self):
        s=next(s for s in self.specs if s['family']=='clahe')
        r=NativeRenderer(s,fast=True);raw=np.random.default_rng(2).integers(3000,10000,(512,640),dtype=np.uint16)
        set_threads(1);a=r(raw);set_threads(8);b=r(raw);set_threads(1)
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()

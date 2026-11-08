import unittest,json
from pathlib import Path
import cv2,numpy as np
from agc_cuda.backend import CudaRenderer,CudaBatch,info
from agc_cuda.reference_methods import Renderer

class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs=json.loads(Path('configs/default.json').read_text())['methods'];cv2.setNumThreads(1)
    def test_reference_parity(self):
        cases=[np.arange(16384,dtype=np.uint16).reshape(128,128),
               np.full((64,64),8000,np.uint16),
               np.random.default_rng(3).integers(3000,12000,(73,61),dtype=np.uint16)]
        for raw in cases:
            for s in self.specs:
                with self.subTest(shape=raw.shape,method=s['name']):
                    r=CudaRenderer(s,width=raw.shape[1],height=raw.shape[0])
                    np.testing.assert_array_equal(r(raw),Renderer(s)(raw));r.close()
    def test_device_timing_and_upload(self):
        r=CudaRenderer(self.specs[0]);raw=np.full((512,640),9000,np.uint16);r.upload(raw)
        self.assertGreater(r.device_time(),0);self.assertGreater(r.device_time_many(),0)
        self.assertEqual(r(raw).shape,raw.shape);r.close()
    def test_batch_equivalence(self):
        s=self.specs[5];batch=CudaBatch(s);single=CudaRenderer(s)
        frames=[np.random.default_rng(i).integers(3000,12000,(512,640),dtype=np.uint16) for i in range(4)]
        for out,frame in zip(batch(frames),frames):np.testing.assert_array_equal(out,single(frame))
        batch.close();single.close()
    def test_invalid_dtype(self):
        r=CudaRenderer(self.specs[0])
        with self.assertRaises(ValueError):r(np.zeros((512,640),np.uint8))
        r.close()

if __name__=='__main__':unittest.main()

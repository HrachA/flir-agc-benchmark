import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from agc_benchmark.methods import Renderer,histogram,quantile,cost
from agc_benchmark.metrics import Reference

class CoreTests(unittest.TestCase):
    def test_histogram_percentiles_match_numpy(self):
        raw=np.random.default_rng(0).integers(1000,10000,(64,64),dtype=np.uint16)
        for p in [0,.1,1,5,50,95,99,99.9,100]:
            self.assertAlmostEqual(quantile(histogram(raw),p),np.percentile(raw,p))

    def test_equal_images(self):
        im=np.random.default_rng(1).integers(0,256,(64,64),dtype=np.uint8)
        m=Reference(im).evaluate(im)
        self.assertAlmostEqual(m['ssim'],1,places=10)
        self.assertEqual(m['mae'],0);self.assertEqual(m['hist_w1_graylevels'],0)
        self.assertTrue(np.isinf(m['psnr_db']))

    def test_global_mappings_monotonic(self):
        raw=np.arange(16384,dtype=np.uint16).reshape(128,128)
        for spec in [{'name':'he','family':'he'},
                     {'name':'phe','family':'plateau','plateau_count':10,'tail_percent':.1},
                     {'name':'lin','family':'linear_hist','tail_percent':1,'midpoint':127}]:
            out=Renderer(spec)(raw)
            self.assertTrue(np.all(np.diff(out.astype(int).ravel())>=0))
            self.assertEqual(out.dtype,np.uint8)

    def test_auto_bright_center_and_shift_invariance(self):
        raw=np.arange(4096,dtype=np.uint16).reshape(64,64)+3000
        render=Renderer({'family':'auto_bright','contrast':8})
        np.testing.assert_array_equal(render(raw),render(raw+100))
        self.assertEqual(int(render(np.full((64,64),5000,np.uint16))[0,0]),127)

    def test_constant_frames_are_finite(self):
        raw=np.full((64,64),6000,np.uint16)
        for family in ['he','he8','plateau','linear_hist','clahe','manual','auto_bright']:
            spec={'family':family,'tail_percent':.1,'plateau_count':10,'midpoint':127,
                  'clip_limit':2,'grid':8,'contrast':8,'brightness':6000}
            out=Renderer(spec)(raw)
            self.assertEqual(out.shape,raw.shape);self.assertEqual(out.dtype,np.uint8)
            self.assertTrue(np.isfinite(Reference(np.full(raw.shape,127,np.uint8)).evaluate(out)['ssim']))

    def test_work_scales_with_pixels(self):
        s={'family':'clahe','name':'x','grid':8}
        a,b=cost(s,640,512),cost(s,1280,1024)
        self.assertEqual(b['work_proxy_at_resolution']-a['histogram_bin_work_proxy'],
                         4*(a['work_proxy_at_resolution']-a['histogram_bin_work_proxy']))

    def test_selection_uses_validation_not_test(self):
        from agc_benchmark.run import aggregate,write_csv
        config={'methods':[{'name':'a','family':'he'},{'name':'b','family':'he'}]}
        rows=[]
        for split,scores in [('val',[.9,.8]),('test',[.1,.99])]:
            for name,ssim in zip(['a','b'],scores):
                rows.append({'split':split,'image_id':'im','sequence_id':split,'method':name,'family':'he',
                             'ssim':ssim,'psnr_db':20,'mae':10,'rmse':15,'correlation':.8,'hist_w1_graylevels':2})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            write_csv(out/'manifest.csv',[{'split':s,'image_id':'im'} for s in ['val','test']])
            write_csv(out/'per_image_metrics.csv',rows)
            _,selected=aggregate(out,config)
            self.assertEqual({r['method'] for r in selected},{'a'})
            write_csv(out/'per_image_metrics.csv',rows[:-1])
            with self.assertRaisesRegex(ValueError,'incomplete'):
                aggregate(out,config)

if __name__=='__main__':unittest.main()

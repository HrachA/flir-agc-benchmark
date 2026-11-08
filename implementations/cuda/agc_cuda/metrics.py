"""Full-frame Gaussian SSIM matches the earlier study's metric convention."""
import cv2
import numpy as np

def blur(x):
    return cv2.GaussianBlur(x, (11,11), 1.5, borderType=cv2.BORDER_REFLECT_101)

class Reference:
    def __init__(self, image):
        self.y = image.astype(np.float64)
        self.mu = blur(self.y)
        self.var = blur(self.y*self.y)-self.mu*self.mu
        self.center = self.y-self.y.mean()
        self.norm = np.sqrt(np.sum(self.center*self.center))
        self.cdf = np.cumsum(np.bincount(image.ravel(),minlength=256))/image.size

    def evaluate(self, pred):
        x, y = pred.astype(np.float64), self.y
        diff = x-y
        mse = np.mean(diff*diff)
        mx = blur(x)
        vx = blur(x*x)-mx*mx
        cov = blur(x*y)-mx*self.mu
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        ssim = np.mean(((2*mx*self.mu+c1)*(2*cov+c2))/
                       ((mx*mx+self.mu*self.mu+c1)*(vx+self.var+c2)+1e-12))
        xc = x-x.mean()
        denom = np.sqrt(np.sum(xc*xc))*self.norm
        corr = float(np.sum(xc*self.center)/denom) if denom>1e-12 else (1.0 if np.array_equal(x,y) else 0.0)
        cdf = np.cumsum(np.bincount(pred.ravel(),minlength=256))/pred.size
        return {'ssim':float(ssim), 'psnr_db':float(10*np.log10(255**2/mse)) if mse else float('inf'),
                'mae':float(np.mean(np.abs(diff))), 'rmse':float(np.sqrt(mse)),
                'correlation':corr, 'hist_w1_graylevels':float(np.sum(np.abs(cdf-self.cdf)))}

import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sklearn.metrics as skm
from skimage.metrics import structural_similarity as compare_ssim
from scipy.fftpack import dctn
from scipy.signal import convolve2d
from scipy.ndimage import sobel, generic_gradient_magnitude


def analysis_MI(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    B = image_ir.astype(np.uint8)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    A = np.reshape(A, -1)
    B = np.reshape(B, -1)
    F = np.reshape(F, -1)
    
    haf = skm.mutual_info_score(A, F)
    hbf = skm.mutual_info_score(B, F)
    return haf + hbf

def analysis_ssim(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    B = image_ir.astype(np.uint8)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.uint8)

    ssim_AF = compare_ssim(A,F)
    ssim_BF = compare_ssim(B,F)

    ssim_mean = (ssim_AF + ssim_BF) * 0.5
    return ssim_mean


def analysis_AG(image):
    img = image.astype(np.float32)
    if len(image.shape) == 2:
        img = img[:,:,np.newaxis]
    h,w,c = img.shape
    g = np.zeros(c)
    for i in range(c):
        image_channel = img[:,:,i]
        [dy, dx] = np.gradient(image_channel)
        s = np.sqrt((np.power(dx,2) + np.power(dy,2))/2); 
        g[i] = np.sum(s) / h / w; 
    val = np.mean(g)
    return val

def analysis_CC(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC

def analysis_VIF(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF


def analysis_SCD(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    # calculates image quality metric value based on THE SUM OF THE CORRELATIONS OF DIFFERENCES (SCD).
    # inputs: A and B are the source images
    # F is the fused image
    # Please site:
    # V. Aslantas and E. Bendes,
    # "A new image quality metric for image fusion: The sum of the correlations of differences,"
    # AEU - International Journal of Electronics and Communications, vol. 69/12, pp. 1890-1896, 2015.
    r = corr2(F - B, A) + corr2(F - A, B)
    return r


def analysis_PSNR(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A)**2))/(m*n)
    MSE_BF = np.sum(np.sum((F - B)**2))/(m*n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255/np.sqrt(MSE))
    return PSNR

def analysis_MSE(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A)**2))/(m*n)
    MSE_BF = np.sum(np.sum((F - B)**2))/(m*n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE


def analysis_SD(image_array):
    image_array = image_array.astype(np.float32)
    if len(image_array.shape) == 2:
        image_array = image_array[:,:,np.newaxis]
    m, n, c = image_array.shape
    SD = np.zeros(c)
    for i in range(c):
        u = np.mean(image_array[:,:,i])
        SD[i] = np.sqrt(np.sum(np.sum((image_array[:,:,i] - u) ** 2)) / (m * n))
        pass
    return SD.mean()


def analysis_EN(image_array):
    if len(image_array.shape) == 2:
        image_array = image_array[:,:,np.newaxis]
    h,w,c = image_array.shape
    entropy = 0
    for i in range(c):
        histogram, bins = np.histogram(image_array[:,:,i].astype(np.uint8), bins=256, range=(0, 255))
        histogram = histogram / float(np.sum(histogram))
        entropy = entropy - np.sum(histogram * np.log2(histogram + 1e-7))
    entropy = entropy / c
    return entropy

def analysis_SF(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(4-scale+1)+1
        win = fspecial_gaussian((N, N), N/5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = convolve2d(ref*ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist*dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref*dist, win, mode='valid') - mu1_mu2

        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12

        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        num += np.sum(np.log10(1+g**2 * sigma1_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1+sigma1_sq/sigma_nsq))
    vifp = num/den
    return vifp




def analysis_FMI(ima, imb, imf, feature, w):
    ima = np.double(ima)
    imb = np.double(imb)
    imf = np.double(imf)

    # Feature Extraction
    if feature == 'none':  # Raw pixels (no feature extraction)
        aFeature = ima
        bFeature = imb
        fFeature = imf
    elif feature == 'gradient':  # Gradient
        aFeature = generic_gradient_magnitude(ima, sobel)
        bFeature = generic_gradient_magnitude(imb, sobel)
        fFeature = generic_gradient_magnitude(imf, sobel)
    elif feature == 'edge':  # Edge
        aFeature = np.double(sobel(ima) > w)
        bFeature = np.double(sobel(imb) > w)
        fFeature = np.double(sobel(imf) > w)
    elif feature == 'dct':  # DCT
        aFeature = dctn(ima, type=2, norm='ortho')
        bFeature = dctn(imb, type=2, norm='ortho')
        fFeature = dctn(imf, type=2, norm='ortho')
    elif feature == 'wavelet':  # Discrete Meyer wavelet
        raise NotImplementedError('Wavelet feature extraction not yet implemented in Python!')
    else:
        raise ValueError(
            "Please specify a feature extraction method among 'gradient', 'edge', 'dct', 'wavelet', or 'none' (raw pixels)!")

    m, n = aFeature.shape
    w = w // 2
    fmi_map = np.ones((m - 2 * w, n - 2 * w))
    pass


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gradient(x,flag):
    x = x.astype(np.float32)
    if flag == 1:
        y = np.concatenate([x, x], axis=1)[:,1:1 + x.shape[1]]
        return y - x
    elif flag == 3:
        y = np.concatenate([x, x], axis=0)[1:1 + x.shape[1],:]
        return y - x

def tf2np(image_tf):
    n,c,h,w = image_tf.size()
    assert n == 1
    if c == 1:
        image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
    else:
        image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    
    return image_np


def image_merge(image_Y_tf,image_YCrCb_tf):
    with torch.no_grad():
        image_merge_tf = image_YCrCb_tf
        image_merge_tf[:,0,:,:] = image_Y_tf
        image_YCrCb_np = (tf2np(torch.clamp(image_merge_tf,min=0.,max=1.)) * 255).astype(np.uint8)
        image_fusion = cv2.cvtColor(image_YCrCb_np, cv2.COLOR_YCR_CB2RGB)
    return image_fusion.astype(np.uint8)


def tf2img(image_tf):
    image_np = tf2np(torch.clamp(image_tf,min=0.,max=1.))
    image_np = (image_np * 255).astype(np.uint8)
    return image_np.astype(np.uint8)


if __name__  == '__main__':
    print('Hello World')
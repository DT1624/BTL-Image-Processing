import numpy as np
import cv2
import math
from skimage.morphology import disk
from skimage.filters.rank import median as rank_median

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255.0
    return img

def mae(original, restored):
    return np.mean(np.abs(original - restored))

def mse(original, restored):
    return np.mean((original - restored)**2)

def add_mixed_noise(img, snr_db, lam):
    P_signal = np.mean(img**2)
    # σ_v^2 = P_signal / (10^(SNR/10))
    sigma_v2 = P_signal / (10**(snr_db/10))
    denom = (1 - lam + 1.0/lam)
    sigma_n2 = sigma_v2/denom
    sigma_n = math.sqrt(sigma_n2)
    h,w = img.shape
    mask = np.random.rand(h,w)
    noise = np.zeros((h,w), dtype=np.float32)
    noise_part1 = np.random.randn(np.sum(mask<(1-lam))) * sigma_n
    noise_part2 = np.random.randn(np.sum(mask>=(1-lam))) * (sigma_n/lam)
    noise[mask<(1-lam)] = noise_part1
    noise[mask>=(1-lam)] = noise_part2
    noisy = img + noise
    noisy = np.clip(noisy,0,1)
    return noisy

def original_filter(noisy_img):
    return noisy_img.copy()

import numpy as np
import cv2
from skimage.morphology import disk, opening, closing, dilation, erosion
from skimage.filters.rank import median as rank_median
from skimage.filters.rank import mean as rank_mean
from skimage.filters import rank
from skimage.util import img_as_ubyte, img_as_float

def rational_filter(noisy_img):
    h,w = noisy_img.shape
    filtered = noisy_img.copy()
    for r in range(1,h-1):
        for c in range(1,w-1):
            x_center = noisy_img[r,c]
            x_left = noisy_img[r,c-1]
            denom = 1.0 + (x_center - x_left)**2
            filtered[r,c] = x_center/denom
    return filtered

def cwm_filter_patch(patch):
    center_val = patch[1,1]
    m = np.median(patch)
    return (center_val*5 + m*1)/6.0

def fir_subfilter(patch, h0):
    return np.sum(patch * h0)

def median_subfilter(patch):
    return np.median(patch)

def mrhf_filter(noisy_img, mode='MRHF1', k=3.2, alpha=[1,-2,1]):
    h,w = noisy_img.shape
    filtered = np.zeros_like(noisy_img)
    h0 = np.array([0.1,0.2,0.1,0.2], dtype=np.float32)
    h1 = np.array([0.2,0.1,0.2,0.1], dtype=np.float32)

    def cwm_val(r,c):
        r1,r2 = max(r-1,0), min(r+1,h-1)
        c1,c2 = max(c-1,0), min(c+1,w-1)
        patch = noisy_img[r1:r2+1,c1:c2+1]
        temp = np.ones((3,3))*patch[(patch.shape[0]//2),(patch.shape[1]//2)]
        temp[0:patch.shape[0],0:patch.shape[1]] = patch
        return cwm_filter_patch(temp)

    def get_phi1(r,c):
        if mode == 'MRHF1':
            pts = []
            if c-1>=0: pts.append(noisy_img[r,c-1])
            else: pts.append(noisy_img[r,c])
            if r-1>=0: pts.append(noisy_img[r-1,c])
            else: pts.append(noisy_img[r,c])
            if c-2>=0: pts.append(noisy_img[r,c-2])
            else: pts.append(noisy_img[r,c])
            if r-2>=0: pts.append(noisy_img[r-2,c])
            else: pts.append(noisy_img[r,c])
            pts = np.array(pts)
            return fir_subfilter(pts, h0[:pts.size])
        elif mode == 'MRHF2':
            line = noisy_img[r,max(c-1,0):min(c+2,w)]
            return median_subfilter(line)
        else:
            diag_pts = []
            if r-1>=0 and c-1>=0: diag_pts.append(noisy_img[r-1,c-1])
            diag_pts.append(noisy_img[r,c])
            if r+1<h and c+1<w: diag_pts.append(noisy_img[r+1,c+1])
            diag_pts = np.array(diag_pts)
            return median_subfilter(diag_pts)

    def get_phi3(r,c):
        if mode == 'MRHF1':
            pts = []
            if c+1<w: pts.append(noisy_img[r,c+1])
            else: pts.append(noisy_img[r,c])
            if r+1<h: pts.append(noisy_img[r+1,c])
            else: pts.append(noisy_img[r,c])
            if c+2<w: pts.append(noisy_img[r,c+2])
            else: pts.append(noisy_img[r,c])
            if r+2<h: pts.append(noisy_img[r+2,c])
            else: pts.append(noisy_img[r,c])
            pts = np.array(pts)
            return fir_subfilter(pts, h1[:pts.size])
        elif mode == 'MRHF2':
            col_values = noisy_img[max(r-1,0):min(r+2,h), c]
            return median_subfilter(col_values)
        else:
            diag_pts = []
            if r-1>=0 and c+1<w: diag_pts.append(noisy_img[r-1,c+1])
            diag_pts.append(noisy_img[r,c])
            if r+1<h and c-1>=0: diag_pts.append(noisy_img[r+1,c-1])
            diag_pts = np.array(diag_pts)
            return median_subfilter(diag_pts)

    alpha1, alpha2, alpha3 = alpha
    for rr in range(h):
        for cc in range(w):
            p1 = get_phi1(rr,cc)
            p2 = cwm_val(rr,cc)
            p3 = get_phi3(rr,cc)
            denominator = k + (p1 - p3)**2
            numerator = alpha1*p1 + alpha2*p2 + alpha3*p3
            y = p2 + numerator/denominator
            filtered[rr,cc] = y

    return filtered

# Thay bridge.png bằng đường dẫn đến ảnh mong muốn
original = read_image('../26.jpg')

lambdas = [0.1, 0.2, 1.0]
#lambdas = [1.0]
SNRs = [3, 9]

methods = {
    'Goc': original_filter,
    'Ti_so': rational_filter,
    'MRHF1': lambda img: mrhf_filter(img, mode='MRHF1'),
    'MRHF2': lambda img: mrhf_filter(img, mode='MRHF2'),
    'MRHF3': lambda img: mrhf_filter(img, mode='MRHF3')
}

results = {snr: {lam: {} for lam in lambdas} for snr in SNRs}

for snr_db in SNRs:
    for lam in lambdas:
        noisy_img = add_mixed_noise(original, snr_db, lam)
        for name, func in methods.items():
            restored = func(noisy_img)
            m_a_e = mae(original, restored)
            m_s_e = mse(original, restored)
            results[snr_db][lam][name] = (m_a_e, m_s_e)

methods_order = ['Goc','Ti_so', 'MRHF1', 'MRHF2', 'MRHF3']

# In 4 bảng:
print("=== MAE với SNR=3 dB ===")
for lam in lambdas:
    print(f"\nλ = {lam}:")
    for m in methods_order:
        mae_8bit = results[3][lam][m][0] * 255
        print(f"{m:6s}: {mae_8bit:.2f}")

print("\n=== MAE với SNR=9 dB ===")
for lam in lambdas:
    print(f"\nλ = {lam}:")
    for m in methods_order:
        mae_8bit = results[9][lam][m][0] * 255
        print(f"{m:6s}: {mae_8bit:.2f}")

print("\n=== MSE với SNR=3 dB ===")
for lam in lambdas:
    print(f"\nλ = {lam}:")
    for m in methods_order:
        mse_8bit = results[3][lam][m][1] * (255 ** 2)
        print(f"{m:6s}: {mse_8bit:.2f}")

print("\n=== MSE với SNR=9 dB ===")
for lam in lambdas:
    print(f"\nλ = {lam}:")
    for m in methods_order:
        mse_8bit = results[9][lam][m][1] * (255 ** 2)
        print(f"{m:6s}: {mse_8bit:.2f}")

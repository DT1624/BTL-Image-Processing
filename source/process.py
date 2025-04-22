# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import wiener


# thêm nhiễu vào hình gốc
def add_impulse_noise_gaussian(original_image, sigma_n=40, lambda_val=0.1):
    m, n = original_image.shape

    noise_main = np.random.normal(0, sigma_n, (m, n))
    noise_impulsive = np.random.normal(0, sigma_n/lambda_val, (m, n))
    mask = np.random.binomial(1, lambda_val, (m, n))
    combined_noise = (1-mask) * noise_main + mask * noise_impulsive
    output = original_image + combined_noise
    output= np.clip(output,0, 255).astype(np.uint8)
    return output

# hiển thị hình ảnh
def display_image_gray(image_path, is_grayscale=True):
    image = cv2.imread(image_path)

    if is_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = add_impulse_noise_gaussian(image)
    cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.uint8)
    # img = add_impulse_noise_gaussian(img)
    return img

# calculate mae loss
def mae_loss(original_image, restored_image):
    return np.mean(np.abs(original_image.astype(np.float32) - restored_image.astype(np.float32)))

# calculate mse loss
def mse_loss(original_image, restored_image):
    return np.mean(np.abs(original_image.astype(np.float32) - restored_image.astype(np.float32)) ** 2)

# MSAMF (Morphological Signal Adaptive Median Filters)
def msamf_filter(noisy_image, kernel_size=3, threshold=30, min_window_size=3, max_window_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(noisy_image, kernel)
    eroded = cv2.erode(noisy_image, kernel)

    morph_gradient = dilated - eroded

    noise_mask = morph_gradient > threshold

    binary_image = (noise_mask > np.mean(noise_mask)).astype(np.uint8)
    eroded_struct = binary_erosion(binary_image, structure=kernel)
    dilated_struct = binary_dilation(binary_image, structure=kernel)
    complexity = np.logical_xor(dilated_struct, eroded_struct).astype(np.uint8)

    output = np.zeros_like(noisy_image)
    pad = max_window_size // 2
    padded = cv2.copyMakeBorder(noisy_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            if noise_mask[i, j]:
                window_size = min_window_size if complexity[i, j] > 0 else max_window_size
                start_i = i + pad - window_size // 2
                start_j = j + pad - window_size // 2
                window = padded[start_i:start_i + window_size, start_j:start_j + window_size]
                output[i, j] = np.median(window)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


# Stack filters
def binary_median_filter(binary_image, kernel_size=3):
    return (cv2.medianBlur(binary_image.astype(np.uint8)*255, kernel_size) // 255).astype(np.uint8)

def stack_filter(noisy_image, levels=255, kernel_size = 7):
    step = 255 // levels
    h, w = noisy_image.shape
    stack = np.zeros((h, w), dtype=np.uint8)

    for i in range(levels):
        threshold = i * step
        binary = (noisy_image >= threshold)
        filtered = binary_median_filter(binary, kernel_size)
        stack += filtered * step
    return stack

# ROMF (Rank-Order Morphological Filters)
def romf_filter(noisy_image, rank='median', kernel_size=5):
    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(noisy_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros_like(noisy_image)

    for i in range(pad, padded.shape[0] - pad):
        for j in range(pad, padded.shape[1] - pad):
            window = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            window_sorted = np.sort(window)
            if rank == 'median':
                output[i - pad, j - pad] = np.median(window_sorted)
            elif rank == 'max':
                output[i - pad, j - pad] = np.max(window_sorted)
            elif rank == 'min':
                output[i - pad, j - pad] = np.min(window_sorted)
            elif rank == 'percentile':
                output[i - pad, j - pad] = np.percentile(window_sorted, 90)
    return output

# rational filters
# def rational_filter(noisy_image, kernel_size=5):
#     return wiener(noisy_image, (kernel_size, kernel_size)).astype(np.uint8)
def rational_filter_freq_domain(img, b=[1, -1, 0.5], a=[1, -0.5, 0.25]):
    """
    Áp dụng Rational Filter trong miền tần số.

    :param img: Ảnh đầu vào (dạng ma trận 2D)
    :param b: Hệ số tử số của rational filter (đa thức trong tử số)
    :param a: Hệ số mẫu số của rational filter (đa thức trong mẫu số)
    :return: Ảnh đã xử lý
    """
    # Chuyển ảnh sang miền tần số bằng FFT
    F = np.fft.fft2(img)

    # Tạo hàm chuyển Rational Filter: H(u, v) = P(u, v) / Q(u, v)
    rows, cols = img.shape
    Y, X = np.fft.fftfreq(rows), np.fft.fftfreq(cols)
    X, Y = np.meshgrid(X, Y)
    Z = X + 1j * Y  # Tạo ma trận Z cho các tần số

    # Tạo hàm chuyển Rational Filter
    H = (b[0] + b[1] * Z + b[2] * Z ** 2) / (a[0] + a[1] * Z + a[2] * Z ** 2)

    # Áp dụng bộ lọc Rational vào ảnh trong miền tần số
    F_filtered = F * H

    # Chuyển ảnh đã lọc về miền không gian
    img_filtered = np.fft.ifft2(F_filtered).real
    return img_filtered

# # FIR Filter (Finite Impulse Response Filter)
def fir_mean_filter(noisy_image, kernel_size=3):
    # kernel có tổng các phần từ bằng 1
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    filtered_image = cv2.filter2D(noisy_image, -1, kernel)
    return filtered_image

def fir_gaussian_filter(noisy_image, kernel_size=3, sigma=2):
    return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigma)

# Median Filter
def median_filter(noisy_image, kernel_size=3):
    filtered_image = cv2.medianBlur(noisy_image, kernel_size)
    return filtered_image

# Subfilter dùng cho MRHF
def fir_subfilter(patch, kernel=None):
    if kernel is None:
        kernel = np.ones(patch, dtype=np.float32) / patch.size
    return np.sum(patch * kernel)

def median_subfilter(patch):
    return np.median(patch)



# MRHF (Median Rational Hybrid Filters)
def mrhf_filter(noisy_image, mode):
    1

if __name__ == "__main__":
    path = "../grayscale/t002.png"
    display_image_gray(path)
    image = read_image(path)

    last_image = add_impulse_noise_gaussian(image)
    print(mse_loss(image, last_image))

    filtered = msamf_filter(last_image)
    print(mse_loss(image, filtered))

    filtered = stack_filter(last_image, levels=255)
    print(mse_loss(image, filtered))

    filtered = romf_filter(last_image)
    print(mse_loss(image, filtered))

    filtered = fir_mean_filter(last_image)
    print(mse_loss(image, filtered))

    filtered = fir_gaussian_filter(last_image)
    print(mse_loss(image, filtered))

    filtered = median_filter(last_image)
    print(mse_loss(image, filtered))

    # # filtered = rational_filter(image)
    # # cv2.imshow("Original", image)
    # print(filtered)
    cv2.imshow("Rational Filtered", filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
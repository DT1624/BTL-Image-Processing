# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation


# Add noise.
def add_impulse_noise_gaussian(original_image, sigma_n=1250, lambda_val=0.1):
    m, n = original_image.shape

    noise_main = np.random.normal(0, np.sqrt(sigma_n), (m, n))
    noise_impulsive = np.random.normal(0, np.sqrt(sigma_n/lambda_val), (m, n))

    # combined_noise = (1-lambda_val) * noise_main + lambda_val  * noise_impulsive
    mask = np.random.binomial(1, lambda_val, (m, n))
    combined_noise = (1-mask) * noise_main + mask * noise_impulsive

    output = original_image + combined_noise
    output= np.clip(output,0, 255).astype(np.uint8)
    return output

# Display image.
def display_image_gray(image_path, is_grayscale=True):
    img = cv2.imread(image_path)
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = add_impulse_noise_gaussian(img)
    cv2.imshow("Image gray", img)

# Read image from path
def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.uint8)
    return img

# Calculate MAE loss
def mae_loss(original_image, restored_image):
    return np.mean(np.abs(original_image.astype(np.float32) - restored_image.astype(np.float32)))

# Calculate MSE loss
def mse_loss(original_image, restored_image):
    return np.mean(np.abs(original_image.astype(np.float32) - restored_image.astype(np.float32)) ** 2)

# MSAMF (Morphological Signal Adaptive Median Filters)
def msamf_filter(noisy_img, kernel_size=3, threshold=10, min_window_size=3, max_window_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(noisy_img, kernel)
    eroded = cv2.erode(noisy_img, kernel)

    morph_gradient = dilated - eroded

    noise_mask = morph_gradient > threshold

    binary_image = (noise_mask > np.mean(noise_mask)).astype(np.uint8)
    eroded_struct = binary_erosion(binary_image, structure=kernel)
    dilated_struct = binary_dilation(binary_image, structure=kernel)
    complexity = np.logical_xor(dilated_struct, eroded_struct).astype(np.uint8)

    output = np.zeros_like(noisy_img)
    pad = max_window_size // 2
    padded = cv2.copyMakeBorder(noisy_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    for i in range(noisy_img.shape[0]):
        for j in range(noisy_img.shape[1]):
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
    stack_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(levels):
        threshold = i * step
        binary = (noisy_image >= threshold)
        filtered_img = binary_median_filter(binary, kernel_size)
        stack_img += filtered_img * step
    return stack_img

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

# Rational filters.
def rational_filter(noisy_image, kernel_size=5, alpha=50):
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(noisy_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    output = np.zeros_like(noisy_image, dtype=np.float32)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            neighborhood = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]

            mean = np.mean(neighborhood)
            variance = np.var(neighborhood)

            center_pixel = padded_image[i, j]

            if variance > 0:
                rational_term = 1 / (1 + alpha * ((center_pixel - mean) ** 2) / variance)
                output[i-pad_size, j-pad_size] = mean + (center_pixel - mean) * rational_term
            else:
                output[i-pad_size, j-pad_size] = center_pixel

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# FIR Filter (Finite Impulse Response Filter)
def fir_mean_filter(noisy_image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    filtered_image = cv2.filter2D(noisy_image, -1, kernel)
    return filtered_image

def fir_gaussian_filter(noisy_image, kernel_size=3, sigma=2):
    return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigma)

# Median Filter
def median_filter(noisy_image, kernel_size=3):
    filtered_image = cv2.medianBlur(noisy_image, kernel_size)
    return filtered_image

# MRHF (Median Rational Hybrid Filters)
def mrhf_filter(original_img, noisy_img, output_file, filter_type='MRHF1', kernel_size=3, h=1.0, k = 0.01, passes=5):
    # CWMF (Center Weighted Median Filter)
    def center_weighted_median(img, kernel=3, center_weight=3):
        pad_size = kernel // 2
        padded_image = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        output_img = np.zeros_like(img, dtype=np.float32)

        for x in range(pad_size, padded_image.shape[0] - pad_size):
            for y in range(pad_size, padded_image.shape[1] - pad_size):
                pixels = [
                    padded_image[x - 1, y],
                    padded_image[x + 1, y],
                    padded_image[x, y - 1],
                    padded_image[x, y + 1],
                    padded_image[x, y]
                ]
                weighted_pixels = pixels[:4] + [pixels[4]] * center_weight
                # calculate median
                output_img[x - pad_size, y - pad_size] = np.median(weighted_pixels).astype(np.float32)

        return output_img


    def fir_subfilter(img, type_filter, kernel=3):
        # Hệ số bộ lọc theo tài liệu
        sqrt_2 = np.sqrt(2)
        h0 = [1 / (2 * (1 + sqrt_2)), 1 / (2 + sqrt_2), 1 / (2 * (1 + sqrt_2)), 1 / (2 + sqrt_2)]
        h1 = [1 / (2 + sqrt_2), 1 / (2 * (1 + sqrt_2)), 1 / (2 + sqrt_2), 1 / (2 * (1 + sqrt_2))]

        coefficients = h0 if type_filter == 'h0' else h1
        pad_size = kernel // 2
        padded_image = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        output_img = np.zeros_like(img, dtype=np.float32)

        for x in range(pad_size, padded_image.shape[0] - pad_size):
            for y in range(pad_size, padded_image.shape[1] - pad_size):
                # Lấy 4 pixel theo hướng tài liệu (h0: 1,2,3,4; h1: 5,6,7,8)
                if type_filter == 'h0':
                    pixels = [
                        padded_image[x - 1, y - 1],  # 1
                        padded_image[x - 1, y],  # 2
                        padded_image[x - 1, y + 1],  # 3
                        padded_image[x, y - 1],  # 4
                        # padded_image[x, y]
                    ]
                else:  # h1
                    pixels = [
                        padded_image[x, y + 1],  # 5
                        padded_image[x + 1, y - 1],  # 6
                        padded_image[x + 1, y],  # 7
                        padded_image[x + 1, y + 1],  # 8
                        # padded_image[x, y]
                    ]
                output_img[x - pad_size, y - pad_size] = np.sum(np.array(pixels) * np.array(coefficients))
        return output_img

    def unidirectional_median_subfilter(img, direction, kernel=3):
        pad_size = kernel // 2
        padded_image = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        output_img = np.zeros_like(img, dtype=np.float32)

        for x in range(pad_size, padded_image.shape[0] - pad_size):
            for y in range(pad_size, padded_image.shape[1] - pad_size):
                if direction == 'central':
                    pixels1 = [
                        padded_image[x - 1, y],
                        padded_image[x + 1, y],
                        padded_image[x, y]
                    ]
                    pixels2 = [
                        padded_image[x, y - 1],
                        padded_image[x, y],
                        padded_image[x, y + 1]  #
                    ]
                    median1 = np.median(pixels1).astype(np.float32)
                    median2 = np.median(pixels2).astype(np.float32)
                    output_img[x - pad_size, y - pad_size] = (median1 + median2) / 2.0
                else: # 'diagonal'
                    pixels1 = [
                        padded_image[x - 1, y - 1],
                        padded_image[x, y],
                        padded_image[x + 1, y + 1]
                    ]
                    pixels2 = [
                        padded_image[x + 1, y - 1],
                        padded_image[x, y],
                        padded_image[x - 1, y + 1]
                    ]
                    median1 = np.median(pixels1).astype(np.float32)
                    median2 = np.median(pixels2).astype(np.float32)
                    output_img[x - pad_size, y - pad_size] = (median1 + median2) / 2.0
        return output_img

    def bidirectional_median_subfilter(img, direction, kernel=3):
        pad_size = kernel // 2
        padded_image = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        output_img = np.zeros_like(img, dtype=np.float32)

        for x in range(pad_size, padded_image.shape[0] - pad_size):
            for y in range(pad_size, padded_image.shape[1] - pad_size):
                if direction == 'central':
                    pixels = [
                        padded_image[x - 1, y],
                        padded_image[x, y - 1],
                        padded_image[x, y],
                        padded_image[x, y + 1],
                        padded_image[x + 1, y]
                    ]
                else:  # diagonal
                    pixels = [
                        padded_image[x - 1, y - 1],
                        padded_image[x - 1, y + 1],
                        padded_image[x, y],
                        padded_image[x + 1, y - 1],
                        padded_image[x + 1, y + 1]
                    ]
                output_img[x - pad_size, y - pad_size] = np.median(pixels).astype(np.float32)
        return output_img

    alpha = np.array([1, -2, 1], dtype=np.float32)
    new_image = noisy_img.copy()

    for x in range(passes):
        # Tính các bộ lọc con
        if filter_type == 'MRHF1':
            phi1 = fir_subfilter(new_image, 'h0', kernel_size)
            phi3 = fir_subfilter(new_image, 'h1', kernel_size)
        elif filter_type == 'MRHF2':
            phi3 = unidirectional_median_subfilter(new_image, 'diagonal', kernel_size)
            phi1 = unidirectional_median_subfilter(new_image, 'central', kernel_size)
        elif filter_type == 'MRHF3':
            phi1 = bidirectional_median_subfilter(new_image, 'central', kernel_size)
            phi3 = bidirectional_median_subfilter(new_image, 'diagonal', kernel_size)
        else:
            raise ValueError("filter_type not is 'MRHF1', 'MRHF2', or 'MRHF3'")

        phi2 = center_weighted_median(new_image, kernel_size, center_weight=3)

        # Output MRHF
        output = np.zeros_like(new_image, dtype=np.float32)
        phi = np.stack([phi1, phi2, phi3], axis=2)

        for i in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                phi_i = phi[i, j]
                numerator = np.sum(alpha * phi_i)
                denominator = h + k * (phi_i[0] - phi_i[2]) ** 2
                output[i, j] = phi_i[1] + numerator / denominator

        output = np.clip(output, 0, 255).astype(np.uint8)
        new_image = output
        with open(output_file, 'a') as file:
            file.write(f"MSE {filter_type} Filter in turn {x}: {mse_loss(original_img, new_image)}\n")
            file.write(f"MAE {filter_type} Filter in turn {x}: {mae_loss(original_img, new_image)}\n")
    return new_image


if __name__ == "__main__":
    file_path = "../grayscale"
    for filename in os.listdir(file_path):
        img_path = os.path.join(file_path, filename)
        original_img = read_image(img_path)
        noisy_img = add_impulse_noise_gaussian(original_img)

        output_file = f"../output/{os.path.splitext(filename)[0]}"
        with open(output_file, 'a') as file:
            file.write(f"MSE noisy image: {mse_loss(original_img, noisy_img)}\n")
            file.write(f"MAE noisy image: {mae_loss(original_img, noisy_img)}\n")

            msamf_img = msamf_filter(noisy_img)
            file.write(f"MSE MSAMF Filter: {mse_loss(original_img, msamf_img)}\n")
            file.write(f"MAE MSAMF Filter: {mae_loss(original_img, msamf_img)}\n")

            stack_img = stack_filter(noisy_img, levels=255)
            file.write(f"MSE Stack Filter: {mse_loss(original_img, stack_img)}\n")
            file.write(f"MAE Stack Filter: {mae_loss(original_img, stack_img)}\n")

            romf_img = romf_filter(noisy_img)
            file.write(f"MSE ROMF Filter: {mse_loss(original_img, romf_img)}\n")
            file.write(f"MAE ROMF Filter: {mae_loss(original_img, romf_img)}\n")

            fir_mean_img = fir_mean_filter(noisy_img)
            file.write(f"MSE FIR Mean Filter: {mse_loss(original_img, fir_mean_img)}\n")
            file.write(f"MAE FIR Mean Filter: {mae_loss(original_img, fir_mean_img)}\n")

            fir_gaussian_img = fir_gaussian_filter(noisy_img)
            file.write(f"MSE FIR Gaussian Filter: {mse_loss(original_img, fir_gaussian_img)}\n")
            file.write(f"MAE FIR Gaussian Filter: {mae_loss(original_img, fir_gaussian_img)}\n")

            median_img = median_filter(noisy_img)
            file.write(f"MSE Median Filter: {mse_loss(original_img, median_img)}\n")
            file.write(f"MAE Median Filter: {mae_loss(original_img, median_img)}\n")

            rational_img = rational_filter(noisy_img)
            file.write(f"MSE Rational Filter: {mse_loss(original_img, rational_img)}\n")
            file.write(f"MAE Rational Filter: {mae_loss(original_img, rational_img)}\n")

        mrhf1_img = mrhf_filter(original_img, noisy_img, output_file, filter_type="MRHF1")
        mrhf2_img = mrhf_filter(original_img, noisy_img, output_file, filter_type="MRHF2")
        mrhf3_img = mrhf_filter(original_img, noisy_img, output_file, filter_type="MRHF3")



# if __name__ == "__main__":
#     path = "../grayscale/t002.png"
#     # display_image_gray(path)
#     image = read_image(path)
#
#     last_image = add_impulse_noise_gaussian(image)
#     print(f"MSE noisy image: {mse_loss(image, last_image)}")
#
#     filtered = msamf_filter(last_image)
#     print(f"MSE MSAMF Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("MSAMF Filtered", filtered)
#
#     filtered = stack_filter(last_image, levels=255)
#     print(f"MSE Stack Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("Stack Filtered", filtered)
#     # cv2.waitKey(0)
#
#     filtered = romf_filter(last_image)
#     print(f"MSE ROMF Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("ROMF Filtered", filtered)
#
#     filtered = fir_mean_filter(last_image)
#     print(f"MSE FIR Mean Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("FIR Mean Filtered", filtered)
#
#     filtered = fir_gaussian_filter(last_image)
#     print(f"MSE FIR Gaussian Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("FIR Gaussian Filtered", filtered)
#
#     filtered = median_filter(last_image)
#     print(f"MSE Median Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("Median Filtered", filtered)
#
#     filtered = rational_filter(last_image)
#     print(f"MSE Rational Filter: {mse_loss(image, filtered)}")
#     cv2.imshow("Rational Filtered", filtered)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
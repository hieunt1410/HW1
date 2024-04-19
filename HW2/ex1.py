import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
# Need to implement here
    def pad(row, padding):
        target = np.concatenate(([row[0]]*padding, row, [row[-1]]*padding), axis=None)
        
        return target
    
    padding = filter_size // 2
    padding_img = np.zeros((padding * 2 + img.shape[0], padding * 2 + img.shape[1]))
    
    for row in range(img.shape[0]):
        if row == 0:
            padding_img[:padding] = pad(img[row], padding) * padding
        elif row == img.shape[0] - 1:
            padding_img[-padding:] = pad(img[row], padding) * padding
        else:
            padding_img[row] = pad(img[row], padding)
        
    return padding_img   

                    
def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
  # Need to implement here
    smoothed_img = np.copy(img)
    
    for i in range(0, img.shape[0] - filter_size + 1):
        for j in range(0, img.shape[1] - filter_size + 1):
            smoothed_img[i:i+filter_size, j:j+filter_size] = np.mean(img[i:i+filter_size, j:j+filter_size])
    
    return smoothed_img

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
  # Need to implement here
    smoothed_img = np.copy(img)
    for i in range(0, img.shape[0] - filter_size + 1):
        for j in range(0, img.shape[1] - filter_size + 1):
            smoothed_img[i:i+filter_size, j:j+filter_size] = np.median(img[i:i+filter_size, j:j+filter_size])
            
    return smoothed_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here
    MAX = 255
    MSE = np.sum(np.square(gt_img - smooth_img)) / (gt_img.shape[0] * gt_img.shape[1])
    
    return 10 * np.log(MAX ** 2 / MSE)



def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png"
    img_gt = "ex1_images/ori_img.png"
    img = read_img(img_noise)
    filter_size = 3

    padded_img = padding_img(img, filter_size=filter_size)
    # show_res(img, padded_img)
    # # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    # show_res(img, mean_smoothed_img)
    # print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    # print('PSNR score of median filter: ', psnr(img, median_smoothed_img))


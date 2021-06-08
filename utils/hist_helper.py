import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np


def get_hists(img, boxs):
    hists = []
    for box in boxs:
        sub_image = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]
        (H, hogImage) = hog(sub_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hist = cv2.calcHist([sub_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hists.append((hogImage, cv2.normalize(hist, hist).flatten()))
    return hists

def calc_color_histogram(im, roi, num_bins=32):
    mask = None
    patch = get_subwindow(im, roi)
    blue_model = cv2.calcHist([patch], [0], mask, [num_bins],  [0,256]).flatten()
    green_model = cv2.calcHist([patch], [1], mask, [num_bins],  [0,256]).flatten()
    red_model = cv2.calcHist([patch], [2], mask, [num_bins],  [0,256]).flatten()

    color_patch = np.concatenate((blue_model, green_model, red_model))
    color_patch = color_patch/np.sum(color_patch)
    return color_patch

def get_subwindow(im, roi):
    _p1 = np.array(range(0, int(roi[2]))).reshape([1, int(roi[2])])
    _p2 = np.array(range(0, int(roi[3]))).reshape([1, int(roi[3])])
    ys = np.floor(roi[0]) + _p1 - np.floor(roi[2] / 2)
    xs = np.floor(roi[1]) + _p2 - np.floor(roi[3] / 2)

    # Check for out-of-bounds coordinates, and set them to the values at the borders
    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[xs > np.size(im, 1) - 1] = np.size(im, 1) - 1
    ys[ys > np.size(im, 0) - 1] = np.size(im, 0) - 1
    xs = xs.astype(int)
    ys = ys.astype(int)
    # extract image
    out1 = im[list(ys[0, :]), :, :]
    out = out1[:, list(xs[0, :]), :]

    return out

def main():
    img = cv2.imread('D:/DATASET/2DMOT2015/test/ADL-Rundle-1/img1/000001.jpg')
    x = 1344
    y = 386
    w = 31
    h = 71

    sub_image = img[y:y + h, x:x + w, :]
    sub_image = cv2.resize(sub_image, (260, 92), cv2.INTER_AREA)

    #hst = calc_color_histogram(img, [x, y, w, h])

    (H, hogImage) = hog(sub_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    #hogImage = hogImage.astype("uint8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    ax1.imshow(sub_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax2.imshow(hogImage, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.show()

#main()

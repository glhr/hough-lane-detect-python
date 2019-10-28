from scipy import signal, ndimage
import numpy as np
from PIL import Image
import cv2

from collections import namedtuple

import matplotlib.pyplot as plt


def do_canny(img, low, high, plot=False):
    print(" === CANNY === ")

    gradient_mag, gradient_ang = calculate_gradients(img)

    nonmax = nonmaxsuppression(gradient_mag, gradient_ang)
    # thresh = thresholding(nonmax, 0.2, 0.3)

    median = np.median(gradient_mag)
    low = np.int(median - median/3)
    high = np.int(median + median/3)

    thresh = thresholding(nonmax, low, high)
    hyst, n = hysteresis(thresh)

    # for debugging
    print("Double thresholding - found",np.count_nonzero(thresh == 100),"weak &",np.count_nonzero(thresh == 255),"strong edges")
    print("Hysteresis - Turned",n,"weak edges into strong edges")

    if plot:
        # plot results
        plt.subplot(321),plt.imshow(img, cmap='gray')
        plt.title('Blurred image'), plt.xticks([]), plt.yticks([])

        plt.subplot(322),plt.imshow(gradient_mag, cmap='gray')
        plt.title('Gradient magnitude'), plt.xticks([]), plt.yticks([])
        # plt.subplot(244),plt.hist(gradient_mag)
        save_array_as_img(gradient_mag, "gradient_mag")

        plt.subplot(323),plt.imshow(nonmax, cmap='gray')
        plt.title('After non-maximum suppression'), plt.xticks([]), plt.yticks([])
        # plt.subplot(246),plt.hist(nonmax)
        save_array_as_img(nonmax, "nonmax")

        plt.subplot(324),plt.imshow(thresh, cmap='gray')
        plt.title('After double thresholding'), plt.xticks([]), plt.yticks([])
        # plt.subplot(248),plt.hist(thresh)
        save_array_as_img(thresh, "thresh")

        plt.subplot(325),plt.imshow(hyst, cmap='gray')
        plt.title('After hysterisis (edge tracking)'), plt.xticks([]), plt.yticks([])
        save_array_as_img(hyst, "hyst")

        plt.show()

    return hyst


def calculate_gradients(img):
    img = img.astype('int32')
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], np.float32)
    edges_x = ndimage.convolve(img, kernel_x, mode='nearest')
    edges_y = ndimage.convolve(img, kernel_y, mode='nearest')

    gradient_mag = np.hypot(edges_x, edges_y)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX)

    gradient_ang = np.arctan2(edges_y, edges_x)

    return gradient_mag, gradient_ang


def nonmaxsuppression(gradient_mag, gradient_ang):

    out = np.zeros_like(gradient_mag, dtype=np.int32) # initialize output array
    gradient_ang = gradient_ang * 180 / np.pi  # convert angles from radians to degrees
    gradient_ang[gradient_ang < 0] += 180  # only keep direction not orientation

    range_tuple = namedtuple('range_tuple', 'low high')
    ranges = {
        0: range_tuple(0, 22.5),
        45: range_tuple(22.5, 67.5),
        90: range_tuple(67.5, 112.5),
        135: range_tuple(112.5, 157.5),
        180: range_tuple(157.5, 180)
    }

    for y in range(1, gradient_mag.shape[0]-1):
        for x in range(1, gradient_mag.shape[1]-1):
            try:
                ang = gradient_ang[y,x]
                val = gradient_mag[y,x]

                if ranges[0].low <= ang < ranges[0].high:
                    neighbours = (gradient_mag[y, x-1],
                                  gradient_mag[y, x+1])

                elif ranges[180].low <= ang <= ranges[180].high:
                    neighbours = (gradient_mag[y, x-1],
                                  gradient_mag[y, x+1])

                elif ranges[90].low <= ang < ranges[90].high:
                    # neighbours = (gradient_mag[y+1, x],
                    #              gradient_mag[y-1, x])
                    neighbours = (255,255) # discard horizontal edges

                elif ranges[45].low <= ang < ranges[45].high:
                    neighbours = (gradient_mag[y+1, x-1],
                                  gradient_mag[y-1, x+1])

                elif ranges[135].low <= ang < ranges[135].high:
                    neighbours = (gradient_mag[y-1, x-1],
                                  gradient_mag[y+1, x+1])

                # if point is a local maxima, keep its value, otherwise set to zero
                if val >= neighbours[0] and val >= neighbours[1]:
                    out[y,x] = val
            except IndexError: # in case point is along image boundary
                pass
    return out


def thresholding(edges, thresh_low, thresh_up):
    weak_val, high_val = 100, 255

    if thresh_up < 1: # relative threshold
        thresh_low = np.int32(thresh_low*edges.max())
        thresh_up = np.int32(thresh_up*edges.max())

    result = edges
    print("Thresholds:",thresh_low,thresh_up)

    weak = (thresh_up > edges) & (edges >= thresh_low)
    result = np.where(edges < thresh_low, 0, result)
    result = np.where(weak, weak_val, result)
    result = np.where(edges >= thresh_up, high_val, result)

    return result


def hysteresis(edges, weak_val=100, strong_val = 255):
    out = np.zeros_like(edges)
    n_changed = 0

    for y in range(1, edges.shape[0]-1):
        for x in range(1, edges.shape[1]-1):
            neighbours = np.array(
                        [edges[y+1, x],
                         edges[y+1, x+1],
                         edges[y+1, x-1],
                         edges[y-1, x],
                         edges[y-1, x+1],
                         edges[y-1, x-1],
                         edges[y, x+1],
                         edges[y, x-1]])
            if edges[y,x] == strong_val:  # keep strong edges
                out[y,x] = strong_val
            elif edges[y,x] == weak_val:  # if weak edge surrounded by at least 1 strong edge, make it strong
                if np.any(neighbours == strong_val):
                    out[y,x] = strong_val
                    n_changed += 1

    return out, n_changed


def save_array_as_img(array,filename):
    # save image to file
    im = Image.fromarray(array)
    im = im.convert("L")
    im.save("canny_output/canny_" + filename + ".png")

    # create and save histogram to file
    hist, edges = np.histogram(array,bins=range(260))
    plt.figure(figsize=[10,8])
    plt.bar(edges[:-1], hist, width = 0.5, color='#0504aa')
    plt.xlim(min(edges), max(edges))
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("canny_output/canny_" + filename + "_histo.png")

    plt.close()

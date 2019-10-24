import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# note: install pillow and matplotlib with the command
# pip install pillow matplotlib

import cv2 # for testing only

# local imports below
from gaussian import do_gaussian
from canny import do_canny
from hough import do_hough_straightline, do_hough_curve

# open test.jpg as grayscale array
img = np.array(Image.open('cam_data/ircam1571825253562129359.png').convert("L"))
#img = np.array(Image.open('canny_output/cannylane_test.jpg').convert("L"))

img = img[250:,:]  # only keep bottom part of image

# downsample image
h,w = img.shape[:2]
desired_w = 240
small_to_large_image_size_ratio = desired_w/w
img = cv2.resize(img,
                   (0,0), # set fx and fy, not the final size
                   fx=small_to_large_image_size_ratio,
                   fy=small_to_large_image_size_ratio,
                   interpolation=cv2.INTER_LINEAR)

blurred = do_gaussian(img) # implement function in gaussian.py
blurred_cv = cv2.GaussianBlur(img, (3,3),0) # OpenCV built-in function, for testing only

edges = do_canny(blurred_cv) # implement function in canny.py
edges_cv = cv2.Canny(blurred_cv, 50, 150) # OpenCV built-in function, for testing only

detected_lines = do_hough_straightline(edges) # implement function in hough.py

# plot results
plt.subplot(221),plt.imshow(img, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blurred, cmap='gray')
plt.title('Blurred image (Gaussian filter OpenCV)'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges_cv, cmap='gray')
plt.title('Canny edge detection (OpenCV)'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(edges, cmap='gray')
plt.title('Canny edge detection (homemade)'), plt.xticks([]), plt.yticks([])
plt.show()

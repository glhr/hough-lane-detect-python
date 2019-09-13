import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# note: install pillow and matplotlib with the command
# pip install pillow matplotlib

import cv2 # for testing only

# local imports below
from gaussian import do_gaussian
from canny import do_canny
from hough import do_hough

# open test.jpg as grayscale array
img = np.array(Image.open('test.jpg').convert("L"))

# blurred = do_gaussian(img) # implement function in gaussian.py
blurred = cv2.GaussianBlur(img, (5,5), 0) # OpenCV built-in function, for testing only

# edges = do_canny(blurred) # implement function in canny.py
edges = cv2.Canny(blurred, 50, 150) # OpenCV built-in function, for testing only

detected_lines = do_hough(edges) # implement function in hough.py

# plot results
plt.subplot(221),plt.imshow(img, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blurred, cmap='gray')
plt.title('Blurred image (Gaussian filter)'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges, cmap='gray')
plt.title('Canny edge detection'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(detected_lines, cmap='gray')
plt.title('Line detection (Hough transform)'), plt.xticks([]), plt.yticks([])
plt.show()

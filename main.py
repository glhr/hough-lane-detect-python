import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# note: install pillow and matplotlib with the command
# pip install pillow matplotlib
import glob
import cv2 # for testing only

# local imports below
from gaussian import do_gaussian
from canny import do_canny
from hough import do_hough_straightline, do_hough_curve

PLOT_INTERMEDIARY = False
PLOT_RESULTS = True

GAUSSIAN_SIZE = 5
CANNY_LOW = 5
CANNY_HIGH = 15
ACCUMULATOR_MAX_AREA = 30
N_LINES = 2

def detect_lane(img_path):
    print('-->',img_path)
    # open image in grayscale
    img = np.array(Image.open(img_path).convert("L"))

    # only keep bottom part of image
    img = img[220:,:]

    # downsample image
    h,w = img.shape[:2]
    desired_w = 250
    small_to_large_image_size_ratio = desired_w/w
    img = cv2.resize(img,
                       (0,0), # set fx and fy, not the final size
                       fx=small_to_large_image_size_ratio,
                       fy=small_to_large_image_size_ratio,
                       interpolation=cv2.INTER_LINEAR)

    cv2.imwrite('gaussian_output/input_'+img_path.split('\\')[-1].split('/')[-1],img)
    blurred = do_gaussian(img) # implement function in gaussian.py
    cv2.imwrite('gaussian_output/output_'+img_path.split('\\')[-1].split('/')[-1],blurred)
    
    blurred_cv = cv2.GaussianBlur(img, (GAUSSIAN_SIZE,GAUSSIAN_SIZE),0) # OpenCV built-in function, for testing only

    edges = do_canny(blurred, CANNY_LOW, CANNY_HIGH, plot=PLOT_INTERMEDIARY) # implement function in canny.py
    edges_cv = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH) # OpenCV built-in function, for testing only

    fig = do_hough_straightline(img,edges,N_LINES,ACCUMULATOR_MAX_AREA,plot=PLOT_INTERMEDIARY) # implement function in hough.py

    if PLOT_INTERMEDIARY:
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

    if PLOT_RESULTS:
        # plot results
        #ax_img = fig.axes[0]
        #ax_img.imshow(img, cmap='gray')
        plt.savefig('results/'+img_path.split('\\')[-1].split('/')[-1],bbox_inches='tight')
        #plt.show()


for path in glob.iglob('cam_data/ir/*.png'):
    detect_lane(path)

# detect_lane('canny_output/cannylane_test.jpg')
#detect_lane('cam_data/ir/ircam1571746678401506290.png')

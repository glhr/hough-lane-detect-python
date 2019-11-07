#!./env/bin/python3

import cv2
import numpy as np
import math
import glob
import numexpr as ne

GAUSSIAN_SIZE = 5 # kernel size
CANNY_LOW = 5
CANNY_HIGH = 15

colors = {
            'white':(255, 255, 255),
            'black':(0,0,0),
            'green':(0, 255, 0),
            'blue':(255,0,0),
            'red':(0,0,255),
            'yellow':(255,255,0)
}

def preprocessing(image):
    # only keep bottom part of image
    img = image[220:,:]

    # downsample image
    h,w = img.shape[:2]
    desired_w = 250
    small_to_large_image_size_ratio = 0.3125
    img = cv2.resize(img,
                       (0,0), # set fx and fy, not the final size
                       fx=small_to_large_image_size_ratio,
                       fy=small_to_large_image_size_ratio,
                       interpolation=cv2.INTER_LINEAR)

    blurred = cv2.GaussianBlur(img, (GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0)
    return blurred


def plot_line(a, b, rho, img, opacity=0.8, color='red'):
    y_max, x_max = img.shape[:2]
    pt1 = (0, int(a * 0 + b))
    pt2 = (x_max, int(a * x_max + b))
    cv2.line(img, pt1, pt2, colors[color], 1, cv2.LINE_AA)
    #print(a, b)
    return img


def is_theta_in_range(theta):
    return (theta < np.deg2rad(-10) and theta > np.deg2rad(-70)) or (theta > np.deg2rad(10) and theta < np.deg2rad(70))

def theta_ranges_from_lane_angle(lane_angle):
    theta_offset = 45
    print('Theta limits:',lane_angle-theta_offset,lane_angle+theta_offset)
    return np.deg2rad(np.concatenate((np.arange(lane_angle-theta_offset,-10),np.arange(10,lane_angle+theta_offset))))

def do_hough_straightline(orig, img, lane_angle, n_lines, max_area, plot=False):
    # Copy edges to the images that will display the results in BGR
    color_edges = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img = img[10:-10][10:-10] # ignore image boundaries
    # print("-------------------------------------")
    max_iterations = 20

    h, w = img.shape
    h_orig, w_orig = orig.shape[:2]
    middle = w / 2
    print("Image center:",middle)
    diag = np.ceil(np.hypot(h, w))
    # print(f"IMG dimensions: {img.shape} max. intensity: {np.max(img)}")

    thetas = theta_ranges_from_lane_angle(lane_angle)
    rhos = np.linspace(-diag, diag, diag * 2.0)

    # print(f"diagonal: {diag}")

    accumulator = np.zeros((np.uint64(2 * diag), len(thetas)), dtype=np.uint64)

    for i in range(0, h):
        for j in range(0, w):
            if img[i, j] > 0:  # if we're on an edge
                rho_calc = ne.evaluate("j * cos(thetas) + i * sin(thetas) + diag")
                for rho_i,rho in enumerate(rho_calc):
                    accumulator[np.uint64(rho_calc[rho_i]), rho_i] += 1  # increment accumulator for this coordinate pair

    n = 1
    iterations = 0

    lane_start = [0,0]
    lane_end = [0,0]
    lane_pos = [0,0]
    lane_side = [0,0]

    while n <= n_lines and iterations < max_iterations:
        # print(iterations)
        # find maximum point in accumulator
        # result = np.where(accumulator == np.max(accumulator))
        # print("max. in accumulator:", np.max(accumulator))
        # maxCoordinates = list(zip(result[0], result[1]))
        # print(maxCoordinates)

        max_index = np.argmax(accumulator)  # 2d index of maximum point in accumulator
        theta_index = np.uint64(max_index % accumulator.shape[1])
        rho_index = np.uint64(max_index / accumulator.shape[1])

        # cv2.circle(accumulator, (rho_index,theta_index), 50, (0,255,0), thickness=5, lineType=8, shift=0)

        ang = thetas[theta_index]
        rho = rhos[rho_index]

        lane_pos[n-1] = (ang > 0)
        a = -(np.cos(ang) / np.sin(ang))
        b = rho / np.sin(ang)
        lane_start[n-1] = ((h - 1) - b) / a
        lane_end[n-1] = -b / a
        lane_side[n-1] = (lane_start[n-1] < middle)

        print(f"- Lane {n}: Cartesion form (ax+b): {a:.2f} * x + {b:.2f}")
        # print(f"\t starting at y = ", lane1_start)
        print(f"- Lane {n}: Theta {np.rad2deg(ang):.2f} - Rho {rho:.2f}")

        # print(f"Hough coordinates: rho {rho:.2f}  theta(rad) {ang:.2f}  theta(deg) {np.rad2deg(ang)}")

        if n == 1:
            color_edges = plot_line(a, b, rho, color_edges, color='green')
            n += 1
        elif n == 2:
            if (lane_side[n-1] != lane_side[n-2]) and ((lane_end[n-1] > lane_end[n-2] and lane_start[n-1] > lane_start[n-2]) or (
                    lane_end[n-1] < lane_end[n-2] and lane_start[n-1] < lane_start[n-2])):
                lane_color = 'blue'
                n += 1
            else:
                lane_color = 'red'
            color_edges = plot_line(a, b, rho, color_edges, color=lane_color)

        prev_ang = ang
        prev_rho = rho

        remove_area = max_area
        for i in range(np.int(rho_index - remove_area), np.int(rho_index + remove_area + 1)):
            try:
                accumulator[i][np.int(theta_index - remove_area):np.int(theta_index + remove_area)] = 0
            except:
                pass

        iterations += 1

    print("Solved in",iterations,"iterations")

    return color_edges


def draw_lanes(image_path):
    image = cv2.imread(image_path)
    image = preprocessing(image)
    edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH, None, 3)
    return do_hough_straightline(image, edges, lane_angle=0, n_lines=2, max_area=10, plot=False)


for path in glob.iglob('cam_data/ir/*.png'):
    cv2.imshow('lanes',draw_lanes(path))
    cv2.waitKey(0)

# detect_lane('canny_output/cannylane_test.jpg')
#detect_lane('cam_data/ir/ircam1571746678401506290.png')

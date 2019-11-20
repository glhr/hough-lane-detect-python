#!./env/bin/python3

import cv2
import numpy as np
import math
import glob
import numexpr as ne
import pandas as pd
from functools import reduce
import operator
import math

HOUGH_OPENCV = True

HIST_EQUALIZATION = False
GAUSSIAN_SIZE = 5 # kernel size
if not HIST_EQUALIZATION:
    CANNY_LOW = 8
    CANNY_HIGH = 20
else:
    CANNY_LOW = 50
    CANNY_HIGH = 100
DOWNSCALING_FACTOR = 0.2
HORIZON = 220
MAX_AREA = 21
LANE_ANGLE = 0
THETA_OFFSET = 70

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
    if HIST_EQUALIZATION:
        image = cv2.equalizeHist(image)
    img = image[HORIZON:,:]

    # downsample image
    h,w = img.shape[:2]
    #desired_w = 150

    small_to_large_image_size_ratio = DOWNSCALING_FACTOR
    img = cv2.resize(img,
                       (0,0), # set fx and fy, not the final size
                       fx=small_to_large_image_size_ratio,
                       fy=small_to_large_image_size_ratio,
                       interpolation=cv2.INTER_LINEAR)

    blurred = cv2.GaussianBlur(img, (GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0)
    return blurred

def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def fill_road_area(img,a,b,downsampling):
    h, w = img.shape[:2]
    x = np.arange(w)

    if downsampling < 1:
        b = (b / downsampling) + HORIZON
        print(b)

    line1 = lambda x: a[0]*x+b[0]
    line2 = lambda x: a[1]*x+b[1]
    y1 = line1(x)
    y2 = line2(x)

    points1 = np.array([(xi, yi) for xi, yi in zip(x, y1) if (0<=xi<w and 0<=yi<h)]).astype(np.int32)
    points2 = np.array([(xi, yi) for xi, yi in zip(x, y2) if (0<=xi<w and 0<=yi<h)]).astype(np.int32)

    minpt = [None, None]
    maxpt = [None, None]
    minpt[0] = min(points1, key = lambda t: t[1])
    maxpt[0] = max(points1, key = lambda t: t[1])
    minpt[1] = min(points2, key = lambda t: t[1])
    maxpt[1] = max(points2, key = lambda t: t[1])

    intersect = get_intersect(minpt[0],maxpt[0],minpt[1],maxpt[1])
    print(intersect)
    if intersect[0] < 0 or intersect[1] < 0 or intersect[0] > w-1 or intersect[1] > h-1:
        points = np.concatenate((points1, points2))
    elif intersect[0] == float("inf") or intersect[1] == float("inf"):
        points = np.concatenate((points1,points2))
    else:
        points = np.concatenate(([maxpt[0]], [maxpt[1]], [intersect]))
        print(points)


    # print(points)
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    # print(center)
    points = np.array(sorted(points, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))

    polynomialgon = img.copy()
    try: cv2.fillPoly(polynomialgon, [points], color=[255,255,255])
    except: cv2.fillPoly(polynomialgon, np.int32([points]), color=[255,255,255])
    return polynomialgon

def plot_line(a, b, rho, img, opacity=0.8, color='red',downsampling=1):
    y_max, x_max = img.shape[:2]
    #a = a * downsampling
    if downsampling < 1:
        b = (b / downsampling) + HORIZON
    pt1 = (0, int(a * 0 + b))
    pt2 = (x_max, int(a * x_max + b))
    cv2.line(img, pt1, pt2, colors[color], 1, cv2.LINE_AA)
    #print(a, b)
    return img

def is_theta_in_range(theta):
    return (theta < np.deg2rad(-10) and theta > np.deg2rad(-70)) or (theta > np.deg2rad(10) and theta < np.deg2rad(70))

def theta_ranges_from_lane_angle(lane_angle):
    if lane_angle == None:
        return np.deg2rad(np.concatenate((np.arange(-70,-10),(np.arange(10,70)))))
    theta_offset = THETA_OFFSET
    # print('Theta limits:',lane_angle-theta_offset,lane_angle+theta_offset)
    return np.deg2rad(np.concatenate((np.arange(lane_angle-theta_offset,-10),np.arange(10,lane_angle+theta_offset))))

def do_hough_straightline(orig, img, lane_angle, n_lines, max_area, plot=False):
    # Copy edges to the images that will display the results in BGR
    color_edges = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_orig = orig
    blank_orig = np.zeros_like(orig)

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

        max_index = np.argmax(accumulator)  # 2d index of maximum point in accumulator
        theta_index = np.uint16(max_index % accumulator.shape[1])
        rho_index = np.uint16(max_index / accumulator.shape[1])

        ang = thetas[theta_index]
        rho = rhos[rho_index]

        lane_pos[n-1] = (ang > 0)
        a = -(np.cos(ang) / np.sin(ang))
        b = rho / np.sin(ang)
        lane_start[n-1] = ((h - 1) - b) / a
        lane_end[n-1] = -b / a
        lane_side[n-1] = (lane_start[n-1] < middle)

        print(f"- Lane {n}: Cartesion form (ax+b): {a:.2f} * x + {b:.2f}")
        print(f"- Lane {n}: Theta {np.rad2deg(ang):.2f} - Rho {rho:.2f}")

        if n == 1:
            color_edges =   plot_line(a, b, rho, color_edges, color='green')
            color_orig =    plot_line(a, b, rho, color_orig, color='green', downsampling=DOWNSCALING_FACTOR)
            blank_orig = plot_line(a, b, rho, blank_orig, color='white', downsampling=DOWNSCALING_FACTOR)
            #accumulator = remove_area_around_max(accumulator,max_area,(rho_index,theta_index))
            n += 1
        elif n == 2:
            if      (lane_side[n-1] != lane_side[n-2]) and \
                    ((lane_end[n-1] > lane_end[n-2] and lane_start[n-1] > lane_start[n-2]) \
                or  (lane_end[n-1] < lane_end[n-2] and lane_start[n-1] < lane_start[n-2])):
                lane_color = 'blue'
                color_orig = plot_line(a, b, rho, color_orig, color=lane_color, downsampling=DOWNSCALING_FACTOR)
                blank_orig = plot_line(a, b, rho, blank_orig, color='white', downsampling=DOWNSCALING_FACTOR)
                n += 1
            else:
                lane_color = 'red'
                #accumulator = remove_area_around_max(accumulator,max_area,(rho_index,theta_index))
            color_edges =   plot_line(a, b, rho, color_edges, color=lane_color)

        for i in range(np.int(rho_index - max_area), np.int(rho_index + max_area + 1)):
            try:
                accumulator[i][np.int(theta_index - max_area):np.int(theta_index + max_area)] = 0
            except:
                pass
        prev_ang = ang
        prev_rho = rho

        iterations += 1

    print("Solved in",iterations,"iterations")

    return color_edges, color_orig, blank_orig[HORIZON:,:]

def do_hough_opencv(orig, img, lane_angle, n_lines):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 10, None, 0, 0)
    color_edges = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_orig = orig
    blank_orig = np.zeros_like(orig)

    rho_prev = 0
    theta_prev = 0

    n = 0
    a_lanes = np.zeros(n_lines)
    b_lanes = np.zeros(n_lines)

    if lines is not None:
        for i in range(0, n_lines):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if theta == 0:
                continue
            print(rho,theta)
            a = -(np.cos(theta) / np.sin(theta))
            b = rho / np.sin(theta)

            # (np.abs(theta - theta_prev) > (np.pi/180)*MAX_AREA and np.abs(rho - rho_prev) > MAX_AREA))
            # if ((lane_angle - THETA_OFFSET < np.rad2deg(theta)) and (np.rad2deg(theta) < lane_angle + THETA_OFFSET)):
            print(rho,theta,lane_angle + THETA_OFFSET,lane_angle - THETA_OFFSET)
            color_edges = plot_line(a, b, rho, color_edges, color='white')
            color_orig = plot_line(a, b, rho, color_orig, color='white', downsampling=DOWNSCALING_FACTOR)
            #blank_orig = plot_line(a, b, rho, blank_orig, color='white', downsampling=DOWNSCALING_FACTOR)
            a_lanes[n] = a
            b_lanes[n] = b
            n += 1

            theta_prev = theta
            rho_prev = rho
    if n == 2:
        polynomialgon = fill_road_area(blank_orig, a=a_lanes, b=b_lanes, downsampling=DOWNSCALING_FACTOR)
    return color_edges, color_orig, blank_orig[HORIZON:, :], polynomialgon[HORIZON:, :]




def draw_lanes(image_path):
    image = cv2.imread(image_path,0)
    image_preprocessed = preprocessing(image)
    edges = cv2.Canny(image_preprocessed, CANNY_LOW, CANNY_HIGH, None, 3)
    if not HOUGH_OPENCV:
        return do_hough_straightline(image, edges, lane_angle=LANE_ANGLE, n_lines=2, max_area=MAX_AREA, plot=False)
    return do_hough_opencv(image, edges, lane_angle=LANE_ANGLE, n_lines=2)


def run_detection():
    for path in glob.iglob('labelbox-generate-data/input/*.png'):
        print(path)
        filename = path.split("/")[-1]
        color_edges, color_orig, blank_orig, roadarea = draw_lanes(path)
        #cv2.imshow('orig',orig)
        if SHOW_DETECT:
            cv2.imshow('original',color_orig)
            cv2.imshow('edges', color_edges)
            cv2.imshow('binary', roadarea)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
        print('cam_data/results_binary/'+filename)
        cv2.imwrite('cam_data/results_binary/'+filename,blank_orig)
        cv2.imwrite('cam_data/results_original/' + filename, color_orig)


def evaluate_results():
    results = []
    for path in glob.iglob('cam_data/results_binary/*.png'):
        print(path)
        filename = path.split("/")[-1]
        input_img = cv2.imread('cam_data/results_original/' + filename)
        result_img = cv2.imread(path)


        if result_img.shape[0] > 600-HORIZON:
            result_img = result_img[HORIZON:,:]
        reference_img = cv2.imread('labelbox-generate-data/color_corrected/labeled_'+filename)
        reference_img = reference_img[220:,:,:]

        CROP_TOP = 100
        reference_img = reference_img[CROP_TOP:,:]
        result_img = result_img[CROP_TOP:,:]

        intersect_img = np.zeros_like(reference_img)

        white = np.array([255, 255, 255])
        grey = np.array([1, 1, 1])
        mask_result = cv2.inRange(result_img, grey, white)
        mask_reference = cv2.inRange(reference_img, grey, white)
        if not mask_reference.any():
            continue
        result_size = np.sum(mask_result > 0)
        reference_size = np.sum(mask_reference > 0)

        intersect = np.logical_and(mask_reference,mask_result)
        intersect_img[intersect] = (255,255,255)
        intersect_size = np.sum(intersect)
        score = intersect_size/result_size
        print("--> Score", str(100*score)+'%')

        result = {'file':filename,
                  'score':score}
        results.append(result)

        if SHOW_EVAL:
            cv2.namedWindow('original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('intersect', cv2.WINDOW_NORMAL)
            cv2.imshow('original', input_img)
            cv2.imshow('intersect',intersect_img)
            cv2.waitKey(0)

    df = pd.DataFrame(results)
    average_score = df["score"].mean()

    print(df)
    print(average_score)
    average_results = {
        'score': average_score,
        'n_files':len(df.index),
        'HOUGH_OPENCV':HOUGH_OPENCV,
        'HIST_EQUALIZATION': HIST_EQUALIZATION,
        'GAUSSIAN_SIZE': GAUSSIAN_SIZE,  # kernel size
        'CANNY_LOW': CANNY_LOW,
        'CANNY_HIGH': CANNY_HIGH,
        'DOWNSCALING_FACTOR': DOWNSCALING_FACTOR,
        'MAX_AREA': MAX_AREA,
        'LANE_ANGLE': None,
        'CROP_TOP': CROP_TOP
    }
    avg_df = pd.DataFrame([average_results])
    print(avg_df)

    with open('result_stats.csv', 'a') as f:
        avg_df.to_csv(f, header=True)
    return


SHOW_DETECT = True
SHOW_EVAL = False

run_detection()
#evaluate_results()
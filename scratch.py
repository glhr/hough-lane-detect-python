import numpy as np
import cv2
from functools import reduce
import operator
import math

img = cv2.imread('/home/nemo/Documents/rob7/hough-lane-detect-python/labelbox-generate-data/input/ircam1573473837535574432.png')
img = img[220:,:]
small_to_large_image_size_ratio = 0.2
img = cv2.resize(img,
                 (0, 0),  # set fx and fy, not the final size
                 fx=small_to_large_image_size_ratio,
                 fy=small_to_large_image_size_ratio,
                 interpolation=cv2.INTER_LINEAR)
h, w = img.shape[:2]

theta = [0,0]
rho = [0,0]
a = [0,0]
b = [0,0]
minpt = [None,None]
maxpt = [None,None]


# theta[0] = 1.0471976
# rho[0] = 61.0
# theta[1] = 1.0297443
# rho[1] = 62.0
theta[0] = 0.715585
rho[0] = 54.0
theta[1] = 1.134464
rho[1] = 27.0

a[0] = -(np.cos(theta[0]) / np.sin(theta[0]))
b[0] = rho[0] / np.sin(theta[0])

a[1] = -(np.cos(theta[1]) / np.sin(theta[1]))
b[1] = rho[1] / np.sin(theta[1])

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

def fill_road_area(img,a,b):
    h, w = img.shape[:2]
    x = np.arange(w)
    line1 = lambda x: a[0]*x+b[0]
    line2 = lambda x: a[1]*x+b[1]
    y1 = line1(x)
    y2 = line2(x)

    points1 = np.array([(xi, yi) for xi, yi in zip(x, y1) if (0<=xi<w and 0<=yi<h)]).astype(np.int32)
    points2 = np.array([(xi, yi) for xi, yi in zip(x, y2) if (0<=xi<w and 0<=yi<h)]).astype(np.int32)

    minpt[0] = min(points1, key = lambda t: t[0])
    maxpt[0] = max(points1, key = lambda t: t[0])
    minpt[1] = min(points2, key = lambda t: t[0])
    maxpt[1] = max(points2, key = lambda t: t[0])

    # print(minpt, maxpt)
    if maxpt[0][0] == w-1 and maxpt[0][1] < h-1:
        corners1 = [(w-1,h-1)]
        use_corners1 = True
    else: use_corners1 = False
    if minpt[1][0] == 0 and minpt[1][1] < h-1:
        corners2 = [(0,h-1)]
        use_corners2 = True
    else: use_corners2 = False

    # points2 = np.flipud(points2)
    if use_corners1 and use_corners2:
        points = np.concatenate((points1, corners1, points2, corners2))
    elif use_corners1:
        points = np.concatenate((points1, corners1, points2))
    elif use_corners2:
        points = np.concatenate((points1, points2, corners2))
    else:
        points = np.concatenate((points1, points2))

    intersect = get_intersect(minpt[0],maxpt[0],minpt[1],maxpt[1])
    print(points)
    if intersect[0] < 0 or intersect[1] < 0:
        points = np.concatenate((points1, points2))
    else:
        points = np.concatenate((maxpt[0], intersect, maxpt[1]))

    print(intersect)
    # print(points)
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    # print(center)
    points = np.array(sorted(points, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))

    polynomialgon = img.copy()
    cv2.fillPoly(polynomialgon, [points], color=[255,255,255])
    return polynomialgon



polynomialgon = fill_road_area(img, a = [a[0],a[1]], b = [b[0],b[1]])
cv2.imshow('road area', polynomialgon)
cv2.waitKey(0)
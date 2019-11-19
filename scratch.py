import numpy as np
import cv2
from functools import reduce
import operator
import math

img = cv2.imread('/home/zacefron/hough-lane-detect-python/labelbox-generate-data/input/ircam1573474107535617302.png')
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
theta[0] = 2.268928
rho[0] = -68.0
theta[1] = 1.134464
rho[1] = 27.0

a[0] = -(np.cos(theta[0]) / np.sin(theta[0]))
b[0] = rho[0] / np.sin(theta[0])

a[1] = -(np.cos(theta[1]) / np.sin(theta[1]))
b[1] = rho[1] / np.sin(theta[1])

x = np.arange(w)
line1 = lambda x: a[0]*x+b[0]
line2 = lambda x: a[1]*x+b[1]
y1 = line1(x)
y2 = line2(x)

points1 = np.array([[[xi, yi]] for xi, yi in zip(x, y1) if (0<=xi<w and 0<=yi<h)]).astype(np.int32)
points2 = np.array([[[xi, yi]] for xi, yi in zip(x, y2) if (0<=xi<w and 0<=yi<h)]).astype(np.int32)

minpt[0] = min(points1, key = lambda t: t[0][0])
maxpt[0] = max(points1, key = lambda t: t[0][0])
minpt[1] = min(points2, key = lambda t: t[0][0])
maxpt[1] = max(points2, key = lambda t: t[0][0])

print(minpt, maxpt)
if maxpt[0][0][0] == w-1 and maxpt[0][0][1] < h-1:
    corners1 = [[[w-1,h-1]]]
    use_corners1 = True
else: use_corners1 = False
if minpt[1][0][0] == 0 and minpt[1][0][1] < h-1:
    corners2 = [[[0,h-1]]]
    use_corners2 = True
else: use_corners2 = False

# points2 = np.flipud(points2)
if use_corners1 and use_corners2:
    points = np.concatenate((minpt, corners1, maxpt, corners2))
elif use_corners1:
    points = np.concatenate((minpt, corners1, maxpt))
elif use_corners2:
    points = np.concatenate((minpt, maxpt, corners2))
else:
    points = np.concatenate((minpt, maxpt))

center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points[0]), [len(points[0])] * 2))
points = np.array(sorted(points[0], key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))

polynomialgon = img.copy()
cv2.fillPoly(polynomialgon, [points], color=[255,255,255])
cv2.imshow('Polygon defined by two polynomials', polynomialgon)
cv2.waitKey(0)
import cv2
import glob
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

initialized = False
for path in glob.iglob('results/ircam*.png'):
    img = cv2.imread(path)

    if not initialized:
        height, width, layers = img.shape
        video = cv2.VideoWriter('video'+timestr+'.mp4',fourcc,3,(width,height))
        initialized = True

    video.write(img)

cv2.destroyAllWindows()
video.release()

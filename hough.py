import numpy as np
import matplotlib.pyplot as plt
import cv2

def do_hough(img):

    img = img[10:-10][10:-10] # ignore image boundaries
    print("-------------------------------------")

    h,w = img.shape
    diag = np.ceil(np.hypot(h,w))
    print(f"IMG dimensions: {img.shape} max. intensity: {np.max(img)}")

    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    rhos = np.linspace(-diag, diag, diag * 2.0)

    print(f"diagonal: {diag}")

    accumulator = np.zeros((np.uint64(2 * diag), len(thetas)), dtype=np.uint64)

    for i in range(0,h):
        for j in range(0,w):
            if img[i,j] > 0:  # if we're on an edge
                for theta_i in range(len(thetas)): # calculate rho for every theta
                    theta = thetas[theta_i]
                    rho = np.round(j * np.cos(theta) + i * np.sin(theta)) + diag
                    # print("point",(i,j),"rho",rho,"theta",theta)
                    rho = np.uint64(rho)

                    accumulator[rho,theta_i] += 1  # increment accumulator for this coordinate pair

    # find maximum point in accumulator

    # result = np.where(accumulator == np.max(accumulator))
    print("max. in accumulator:", np.max(accumulator))
    # maxCoordinates = list(zip(result[0], result[1]))
    # print(maxCoordinates)

    max_index = np.argmax(accumulator) # 2d index of maximum point in accumulator
    ang = thetas[np.uint64(max_index % accumulator.shape[1])]
    rho = rhos[np.uint64(max_index / accumulator.shape[1])]

    print(f"Hough coordinates: rho {rho:.2f}  theta(rad) {ang:.2f}  theta(deg) {np.rad2deg(ang)}")

    # Plotting

    def forceAspect(ax,aspect,h = None,w = None):
        im = ax.get_images()
        try:
            extent = im[0].get_extent()
        except IndexError:
            extent = [h,0,w,0]
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


    fig = plt.figure()

    ax_img = fig.add_subplot(131) # original image
    ax_img.imshow(img, cmap='gray')

    ax_acc = fig.add_subplot(132) # accumulator
    ax_acc.imshow(accumulator, cmap='gray')
    forceAspect(ax_acc,2)

    ax_res = fig.add_subplot(133) # estimated line

    ax_res.set_ylim(-h,h)
    ax_res.set_xlim(-w,w)

    a = -(np.cos(ang)/np.sin(ang))
    b = rho/np.sin(ang)

    print(f"Cartesion form (ax+b): {a:.2f} * x + {b:.2f}")

    x_vals = np.int64(np.array(ax_res.get_xlim()))
    y_vals = np.int64(b + a * x_vals)

    img = cv2.cvtColor(np.float32(img),cv2.COLOR_GRAY2RGB)

    cv2.line(img, (x_vals[0], y_vals[0]), (x_vals[-1], y_vals[-1]), (0,255,255), thickness=1)
    ax_res.imshow(img)

    ax_res.invert_yaxis()
    plt.show()

    return None

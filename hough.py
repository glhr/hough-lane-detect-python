import numpy as np
import matplotlib.pyplot as plt
import cv2
import progressbar


def forceAspect(ax,aspect):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*aspect)


def plot_accumulator(accumulator):
    # Create the x, y, and z coordinate arrays.  We use
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    x = np.arange(accumulator.shape[0])[:, None, None]
    y = np.arange(accumulator.shape[1])[None, :, None]
    z = np.arange(accumulator.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)
    c = np.tile(accumulator.ravel()[:, None], [1, 3])

    ax = plt.axes(projection='3d')
    ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=accumulator.ravel());
    plt.show()


def plot_curve(img,k,beta,v,ax_img):
    h,w = img.shape
    x_vals = np.linspace(-w,w,2*w)
    y_vals = k/x_vals + beta*x_vals + v
    ax_img.plot(x_vals,y_vals,'k',color='firebrick',alpha=0.5)
    return ax_img

def do_hough_straightline(img):

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


    # Plotting

    fig = plt.figure()

    ax_img = fig.add_subplot(131) # original image
    ax_img.imshow(img, cmap='gray')

    ax_acc = fig.add_subplot(132) # accumulator
    ax_acc.imshow(accumulator, cmap='gray')
    forceAspect(ax_acc,2)

    ax_res = fig.add_subplot(133) # estimated line

    ax_res.set_ylim(-h,h)
    ax_res.set_xlim(-w,w)

    img = cv2.cvtColor(np.float32(img),cv2.COLOR_GRAY2RGB)

    cv2.imwrite("accumulator.png",accumulator)

    for i in range(2):
        # find maximum point in accumulator

        # result = np.where(accumulator == np.max(accumulator))
        print("max. in accumulator:", np.max(accumulator))
        # maxCoordinates = list(zip(result[0], result[1]))
        # print(maxCoordinates)

        max_index = np.argmax(accumulator) # 2d index of maximum point in accumulator
        theta_index = np.uint64(max_index % accumulator.shape[1])
        rho_index = np.uint64(max_index / accumulator.shape[1])

        ang = thetas[theta_index]
        rho = rhos[rho_index]

        print(f"Hough coordinates: rho {rho:.2f}  theta(rad) {ang:.2f}  theta(deg) {np.rad2deg(ang)}")

        a = -(np.cos(ang)/np.sin(ang))
        b = rho/np.sin(ang)

        print(f"Cartesion form (ax+b): {a:.2f} * x + {b:.2f}")

        x_vals = np.int64(np.array(ax_res.get_xlim()))
        y_vals = np.int64(b + a * x_vals)



        cv2.line(img, (x_vals[0], y_vals[0]), (x_vals[-1], y_vals[-1]), (0,255,255), thickness=1)

        accumulator[rho_index][theta_index] = 0

    ax_res.imshow(img)
    ax_res.invert_yaxis()
    plt.show()

    return None


def do_hough_curve(img):

    N_MAX = 4

    print("-------------------------------------")
    img = np.flipud(img) # vertical flip
    h,w = img.shape
    diag = np.ceil(np.hypot(h,w))
    print(f"IMG dimensions: {img.shape} max. intensity: {np.max(img)}")

    beta_min = 0
    beta_max = 10
    k_min = -2000
    k_max = 0
    betas = np.linspace(0,3,21)
    # betas = [2,3]
    # print(betas)
    ks = np.linspace(k_min,k_max,101)
    # ks = [-1000,-1200,-1500]

    vs = np.arange(0,2*h+1)
    accumulator = np.zeros((len(betas),len(ks),len(vs)), dtype=np.uint64)

    for y in progressbar.progressbar(range(1,h-1)):
        for x in range(1,w-1):
            if img[y,x] > 0:  # if we're on an edge
                # print(i,j)
                for beta_i in range(len(betas)):
                    for k_i in range(len(ks)):
                        v = np.int64(np.round(y - ks[k_i]/x - betas[beta_i]*x))
                        # print(f"x:{x} | y:{y} | beta:{betas[beta_i]} | k:{ks[k_i]} | v:{v}")
                        if v >= -h and v < h:
                            accumulator[beta_i,k_i,v+h] += 1  # increment accumulator for this coordinate pair



    fig = plt.figure()

    ax_img = fig.add_subplot(111)  # original image
    ax_img.imshow(np.flipud(img),cmap='gray',extent=[0, w, 0, h])


    for i in range(N_MAX):
        max = np.unravel_index(accumulator.argmax(), accumulator.shape)
        print(np.max(accumulator),"at",max)
        print("k",ks[max[1]]," | beta",betas[max[0]]," | v",vs[max[2]])

        # plot_accumulator(accumulator)
        ax_img = plot_curve(np.flipud(img),ks[max[1]],betas[max[0]],max[2]-h,ax_img)

        accumulator[max[0]][max[1]][max[2]] = 0


    ax_img.set_ylim(0,h)
    ax_img.set_xlim(0,w)
    forceAspect(ax_img,0.5)

    # plt.savefig('curve.png')
    # ax_img.invert_yaxis()
    plt.show()


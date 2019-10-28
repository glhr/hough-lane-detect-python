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

def plot_line(a,b,ax_res,opacity=0.8,color='firebrick'):
    x_vals = np.int64(np.array(ax_res.get_xlim()))
    y_vals = np.int64(b + a * x_vals)
    ax_res.plot(x_vals,y_vals,'k',color=color,alpha=opacity,linewidth=3)
    return ax_res

def plot_curve(img,k,beta,v,ax_img):
    h,w = img.shape
    x_vals = np.linspace(-w,w,2*w)
    y_vals = k/x_vals + beta*x_vals + v
    ax_img.plot(x_vals,y_vals,'k',color='firebrick',alpha=0.5)
    return ax_img

def is_theta_in_range(theta):
	return (theta < np.deg2rad(-20) and theta > np.deg2rad(-65)) or (theta > np.deg2rad(20) and theta < np.deg2rad(65))

def do_hough_straightline(orig,img,n_lines,max_area,plot=False):

    #img = img[10:-10][10:-10] # ignore image boundaries
    #print("-------------------------------------")
    max_iterations = 5

    h,w = img.shape
    h_orig,w_orig = orig.shape
    middle = w/2
    diag = np.ceil(np.hypot(h,w))
    #print(f"IMG dimensions: {img.shape} max. intensity: {np.max(img)}")

    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    rhos = np.linspace(-diag, diag, diag * 2.0)

    #print(f"diagonal: {diag}")

    accumulator = np.zeros((np.uint64(2 * diag), len(thetas)), dtype=np.uint64)

    for i in range(0,h):
        for j in range(0,w):
            if img[i,j] > 0:  # if we're on an edge
                for theta_i in range(len(thetas)): # calculate rho for every theta
                    theta = thetas[theta_i]
                    rho = np.round(j * np.cos(theta) + i * np.sin(theta)) + diag
                    # print("point",(i,j),"rho",rho,"theta",theta)
                    rho = np.uint64(rho)
                    if is_theta_in_range(theta):
                        accumulator[rho,theta_i] += 1  # increment accumulator for this coordinate pair


    # Plotting

    if plot:
        fig1 = plt.figure()

        ax_acc = fig1.add_subplot(111) # accumulator
        ax_acc.imshow(accumulator, cmap='gray')
        forceAspect(ax_acc,0.5)

    fig2 = plt.figure()

    ax_img = fig2.add_subplot(211) # original image
    ax_img.set_ylim(-h_orig,h_orig)
    ax_img.set_xlim(-w_orig,w_orig)

    ax_res = fig2.add_subplot(212) # estimated line
    ax_res.set_ylim(-h,h)
    ax_res.set_xlim(-w,w)

    # img = cv2.cvtColor(np.float32(img),cv2.COLOR_GRAY2RGB)
    ax_res.imshow(np.flipud(img),cmap='gray',extent=[0, w, 0, h])
    ax_img.imshow(np.flipud(orig),cmap='gray',extent=[0, w_orig, 0, h_orig])

    n = 1
    iterations = 0
    while n <= n_lines and iterations < max_iterations:
        print(iterations)
        # find maximum point in accumulator
        # result = np.where(accumulator == np.max(accumulator))
        #print("max. in accumulator:", np.max(accumulator))
        # maxCoordinates = list(zip(result[0], result[1]))
        # print(maxCoordinates)

        max_index = np.argmax(accumulator) # 2d index of maximum point in accumulator
        theta_index = np.uint64(max_index % accumulator.shape[1])
        rho_index = np.uint64(max_index / accumulator.shape[1])

        ang = thetas[theta_index]
        rho = rhos[rho_index]

        #print(f"Hough coordinates: rho {rho:.2f}  theta(rad) {ang:.2f}  theta(deg) {np.rad2deg(ang)}")
	
        
        if n == 1:
            lane1_pos = (ang > 0)
            a = -(np.cos(ang)/np.sin(ang))
            b = rho/np.sin(ang)
            lane1_start = ((h-1) - b)/a
            lane1_end = -b/a
            lane1_side = (lane1_start < middle)
            print(f"- Lane 1: Cartesion form (ax+b): {a:.2f} * x + {b:.2f}")
            print(f"\t starting at y = ", lane1_start)
            plot_line(a,b,ax_res,color='red')
            plot_line(a,b,ax_img,opacity=0.3,color='red')
            n += 1
        elif n == 2:
            lane2_pos = (ang > 0)
            a = -(np.cos(ang)/np.sin(ang))
            b = rho/np.sin(ang)
            lane2_start = ((h-1) - b)/a
            lane2_end = -b/a
            lane2_side = (lane2_start < middle)
            if (lane1_side != lane2_side) and ((lane2_end > lane1_end and lane2_start > lane1_start) or (lane2_end < lane1_end and lane2_start < lane1_start)):
                print(f"- Lane 2: Cartesion form (ax+b): {a:.2f} * x + {b:.2f}")
                print(f"\t starting at y = ", lane2_start)
                plot_line(a,b,ax_res,color='blue')
                plot_line(a,b,ax_img,opacity=0.3,color='blue')
                n += 1

        prev_ang = ang
        prev_rho = rho

        remove_area = max_area
        for i in range(np.int(rho_index-remove_area),np.int(rho_index+remove_area+1)):
            accumulator[i][np.int(theta_index-remove_area):np.int(theta_index+remove_area)] = 0
        
        iterations += 1

    ax_res.set_ylim(0,h)
    ax_res.set_xlim(0,w)
    ax_res.invert_yaxis()

    ax_img.set_ylim(0,h_orig)
    ax_img.set_xlim(0,w_orig)
    ax_img.invert_yaxis()

    if plot:
        cv2.imwrite("accumulator.png",accumulator)
        plt.show()

    return fig2


def do_hough_curve(orig,img):

    N_MAX = 10

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


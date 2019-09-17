import numpy as np


def do_gaussian(src):
    # gaussian filtering here

    dst = src.copy()

    height = src.shape[0]
    width = src.shape[1]
    sigma = 3
    pi = np.pi
    sum = 0 #for normalizing

    kernelSize = 5
    kernel=np.zeros((kernelSize,kernelSize))


    #kernel
    for c in np.arange(0, kernelSize):
        for r in np.arange(0, kernelSize):
         kernel[c][r] = np.exp(-0.5*(np.power((c-kernelSize/2)/sigma, 2)+np.power((r-kernelSize/2)/sigma, 2.0)))/(2*pi*sigma*sigma)
         sum = sum + kernel[c][r]

    #normalize kernel
    for a in np.arange(0, kernelSize):
        for b in np.arange(0, kernelSize):
           kernel[a][b] = kernel[a][b]/sum

    padded = np.pad(src, pad_width=2, mode='symmetric')
    height = padded.shape[0]
    width = padded.shape[1]

    #convolution
    for i in np.arange(2, height-2):
        for j in np.arange(2, width-2):        
            summ = 0
            for k in np.arange(-2, 3):
                for l in np.arange(-2, 3):
                    a = padded.item(i+k, j+l) #access pixel values

                    p = kernel[2+k, 2+l]
                    summ = summ + (p * a)
            b = summ
            dst.itemset((i-2,j-2), summ) #change pixel value

    return dst

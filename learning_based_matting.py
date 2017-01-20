import numpy as np
import scipy.sparse

from numpy.lib.stride_tricks import as_strided


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def learning_based_matte(img, trimap, c=800, mylambda=0.0000001):
    foreground = trimap == 255
    background = trimap == 0
    mask = np.zeros(trimap.shape)
    mask[foreground] = 1
    mask[background] = -1

    L = getLapFast(img, mask, mylambda)
    C = getC(mask, c)
    alpha = solveQurdOpt(L, C, mask)
    return alpha


def getLapFast(img, mask, mylambda=0.0000001, win_rad=1):

    w_s = win_rad*2 +1
    win_size = (w_s)**2
    img = img/255
    h, w, c = img.shape
    indsM = np.reshape(np.arange(h*w), (h, w))
    ravelImg = img.reshape(h*w, c)

    scribble_mask = mask != 0
    numPix4Training = np.sum(1-scribble_mask[win_rad:-win_rad, win_rad:-win_rad])

    numNonzeroValue = numPix4Training*win_size**2

    row_inds = np.zeros(numNonzeroValue)
    col_inds = np.zeros(numNonzeroValue)
    vals = np.zeros(numNonzeroValue)

    win_indsMat = rolling_block(indsM, block=(w_s, w_s))
    win_indsMat = win_indsMat.reshape(h - 2*win_rad, w - 2*win_rad, win_size)

    t = 0
    # Repeat on each legal pixel
    # Cannot fully vectorize since the size of win_inds varies by the number of unknown pixels
    # If we preform the computation for all pixels in fully vectorised form it's slower.
    for i in range(h - 2*win_rad):
        win_inds = win_indsMat[i, :]
        win_inds = win_inds[np.logical_not(scribble_mask[i+win_rad, win_rad:w-win_rad])]
        winI = ravelImg[win_inds]
        m = winI.shape[0]
        winI = np.concatenate((winI, np.ones((m, win_size, 1))), axis =2)
        I = np.tile(np.eye(win_size), (m, 1, 1))
        I[:, -1, -1] = 0
        winITProd = np.einsum('...ij,...kj ->...ik', winI, winI)
        fenmu = winITProd + mylambda*I
        invFenmu = np.linalg.inv(fenmu)
        F = np.einsum('...ij,...jk->...ik', winITProd, invFenmu )
        I_F = np.eye(win_size) - F
        lapcoeff = np.einsum('...ji,...jk->...ik', I_F, I_F )

        vals[t: t+(win_size**2)*m] = lapcoeff.ravel()
        row_inds[t:t+(win_size**2)*m] = np.repeat(win_inds, win_size).ravel()
        col_inds[t:t+(win_size**2)*m] = np.tile(win_inds, win_size).ravel()
        t = t+(win_size**2)*m
    L = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(h*w, h*w))
    return L


def solveQurdOpt(L, C, alpha_star):
    mylambda = 1e-6
    D = scipy.sparse.eye(L.shape[0])

    alpha = scipy.sparse.linalg.spsolve(L + C + D*mylambda, C @ alpha_star.ravel())
    alpha = np.reshape(alpha, alpha_star.shape)

    # if alpha value of labelled pixels are -1 and 1, the resulting alpha are
    # within [-1 1], then the alpha results need to be mapped to [0 1]
    if np.min(alpha_star.ravel()) == -1:
        alpha = alpha*0.5+0.5
    alpha = np.maximum(np.minimum(alpha, 1), 0)
    return alpha


def getC(mask, c=800):
    scribble_mask = (mask != 0).astype(int)
    numPix = np.prod(mask.shape)
    C = scipy.sparse.diags(c * scribble_mask.ravel())
    return C


def main():
    img = scipy.misc.imread("troll.png")
    trimap = scipy.misc.imread("trollTrimap.png", flatten='True')

    alpha = learning_based_matte(img, trimap)
    scipy.misc.imsave('trollAlpha.png', alpha)
    # plt.imshow(alpha, cmap = 'gray')
    # plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.misc
    main()

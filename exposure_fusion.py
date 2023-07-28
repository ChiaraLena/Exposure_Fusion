import torch
import kornia
import torch.nn.functional as F
import math


pyramid_filter = kornia.filters.kernels.get_gaussian_kernel1d(5, 1).unsqueeze(0)


def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)

    # start with low pass residual
    R = pyr[nlev - 1]
    for l in range(nlev - 2, -1, -1):
        odd = (2 * R.shape[2] - pyr[l].shape[2], 2 * R.shape[3] - pyr[l].shape[3])
        R = pyr[l] + upsample(R, odd, pyramid_filter)
    return R


def exposure_fusion(I, contrast_parm, sat_parm,wexp_parm):
    r = I.shape[2]
    c = I.shape[3]
    N = I.shape[0]

    W = torch.ones((N, 1, r, c))


    # compute the measures and combines them into a weight map
    if (contrast_parm > 0):
        W = torch.mul(W, torch.pow(contrast(I), contrast_parm))

    if (sat_parm > 0):
        W = torch.mul(W, torch.pow(saturation(I), sat_parm))

    if (wexp_parm > 0):
        W = torch.mul(W, torch.pow(well_exposedness(I), wexp_parm))


    # normalize weights: make sure that weights sum to one for each pixel
    W = W + 1e-12  # avoids division by zero
    W = torch.div(W, sum(W, 0).repeat(N, 1, 1, 1))

    # create empty pyramid
    nlev = math.floor(math.log(min(r, c)) / math.log(2))
    pyr = kornia.geometry.transform.build_pyramid(torch.zeros(1, 3, r, c), nlev, border_type='reflect',
                                                  align_corners=False)

    pyrW = kornia.geometry.transform.build_pyramid(W, nlev, border_type='reflect', align_corners=False)


    # multiresolution blending
    pyrI = laplacian_pyramid(I[:, :, :, :], nlev)


    # blend
    for n in range(N):
        for l in range(nlev):
            w = pyrW[l][n, :, :, :].repeat(1, 3, 1, 1)
            pyr[l] = pyr[l] + w * pyrI[l][n, :, :, :]

    R = reconstruct_laplacian_pyramid(pyr)

    R = torch.clip(R, 0, 1)
    return R


def contrast(I):
    h = kornia.filters.get_laplacian_kernel2d(3)  # laplacian filter
    h = h.unsqueeze(dim=0)
    N = I.shape[0]
    C = torch.zeros(N, 1, I.shape[2], I.shape[3])

    for i in range(N):
        mono = kornia.color.rgb_to_grayscale(I[i, :, :, :])
        mono = torch.unsqueeze(mono, dim=1)
        C[i, :, :, :] = abs(kornia.filters.filter2d(mono, h, 'replicate'))
    return C


def saturation(I):
    N = I.shape[0]
    C = torch.zeros(N, 1, I.shape[2], I.shape[3])

    for i in range(N): # saturation is computed as the standard deviation of the color channels
        R = I[i, 0, :, :]
        G = I[i, 1, :, :]
        B = I[i, 2, :, :]
        mu = torch.div((R + G + B), 3)
        C[i, :, :, :] = torch.sqrt((torch.pow((R - mu), 2) + torch.pow((G - mu), 2) + torch.pow((B - mu), 2)) / 3)
    return C


def well_exposedness(I):
    sig = .2
    N = I.shape[0]
    C = torch.zeros(N, 1, I.shape[2], I.shape[3])

    for i in range(N):
        R = torch.exp(-.5 * torch.pow((I[i, 0, :, :] - .5), 2) / pow(sig, 2))
        G = torch.exp(-.5 * torch.pow((I[i, 1, :, :] - .5), 2) / pow(sig, 2))
        B = torch.exp(-.5 * torch.pow((I[i, 2, :, :] - .5), 2) / pow(sig, 2))
        C[i, :, :, :] = R * G * B
    return C


def upsample(I, odd, filter):
    # increase resolution
    I = torch.nn.functional.pad(I, (1, 1, 1, 1), 'replicate')  # pad the image with a 1-pixel border
    N = I.shape[0]
    r = 2 * I.shape[2]
    c = 2 * I.shape[3]
    k = I.shape[1]
    R = torch.zeros(N, k, r, c)
    R[:, :, :r:2, :c:2] = 4 * I  # increase size 2 times; the padding is now 2 pixels wide


    # interpolate, convolve with separable filter
    R = kornia.filters.filter2d_separable(R, filter, filter, 'reflect')


    # remove the border
    R = R[:, :, 2:r - 2 - odd[0], 2:c - 2 - odd[1]]
    return R


def downsample(I, filter):
    # increase resolution
    r = I.shape[2]
    c = I.shape[3]


    # interpolate, convolve with separable filter
    R = kornia.filters.filter2d_separable(I, filter, filter, 'reflect')


    # remove the border
    R = R[:, :, 1:r:2, 1:c:2]
    return R


def laplacian_pyramid(I, nlev):
    pyr = []
    J = torch.clone(I)

    for l in range(nlev - 1):

        # apply low pass filter, and downsample
        I = downsample(J, pyramid_filter)


        # for each dimension, check if the upsampled version has to be odd in each level,
        # store difference between image and upsampled low pass version
        odd = (2 * I.shape[2] - J.shape[2], 2 * I.shape[3] - J.shape[3])
        pyrup_J = upsample(I, odd, pyramid_filter)
        pyr.append(J - pyrup_J)
        J = I  # continue with low pass image

    pyr.append(J)
    return pyr










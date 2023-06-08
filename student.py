'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

import matplotlib.pyplot as plt
import cv2


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###

    # Generate initial small segmentation using Felzenszwalb algorithm
    segments_fz = skimage.segmentation.felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)

    # print(segments_fz.shape)  # (848, 947)
    # print(len(np.unique(segments_fz)))
    # # plot the segmentation area of each segment
    # plt.imshow(segments_fz)
    # plt.savefig('felzenszwalb.png')
    # plt.show()

    # Add segment labels as a 4th channel to the image
    im_orig = np.dstack((im_orig, segments_fz))
    return im_orig


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    return np.minimum(r1['hist_c'], r2['hist_c']).sum()

def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    return np.minimum(r1['hist_t'], r2['hist_t']).sum()

def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    return 1.0 - (r1['size'] + r2['size']) / imsize

def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    # Calculate the tight bounding box around r1 and r2
    bbij = ((max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])))

    return 1.0 - (bbij - r1['size'] - r2['size']) / imsize

def calc_sim(r1, r2, imsize):
    # print("4",sim_colour(r1, r2), sim_texture(r1, r2), sim_size(r1, r2, imsize), sim_fill(r1, r2, imsize))
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


def calc_colour_hist(img):  # 2d regional HSV
    """
    Task 2.5.1
    Calculate colour histogram for each region
    The size of output histogram will be BINS * COLOUR_CHANNELS(3)   # (75,)
    Number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###

    # Calculate colour histogram for each channel and concatenate them
    for channel in range(3):
        hist_ch = np.histogram(img[:, channel], BINS, (0, 1))[0]
        hist = np.hstack((hist, hist_ch))
    # print("1",np.linalg.norm(hist, 1))
    # print("2",len(img))

    # Normalize the histogram using the L1 norm, for the final combination of similarity measures
    hist /= np.linalg.norm(hist, 1) + np.finfo(float).eps
    # print("3",np.linalg.norm(hist, 1))

    return hist


def calc_texture_gradient(img):  # img: 3d with 3 channels
    """
    Task 2.5.2
    Calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    For 8 orientations, but we will use LBP instead.
    Output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###

    for channel in range(3):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(img[:, :, channel], 8, 1.0)

    # print(np.max(ret[:,:,0]), np.max(ret[:,:,1]), np.max(ret[:,:,2]))
    # print(np.min(ret[:,:,0]), np.min(ret[:,:,1]), np.min(ret[:,:,2]))

    # # plot the texture gradient
    # plt.imshow(ret[:,:,0])
    # plt.savefig('texture_gradient.png')
    # plt.show()

    return ret

def calc_texture_hist(img):  # 2d regional texture gradient
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    The size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)   # (240,)??????
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###

    for channel in range(3):
        hist_ch = np.histogram(img[:, channel], BINS)[0]
        hist = np.hstack((hist, hist_ch))

    # Normalize the histogram using the L1 norm
    hist /= np.linalg.norm(hist, 1) + np.finfo(float).eps

    return hist


def extract_regions(img):  # img: 3d with 4 channels
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}  # regions {0:{}, 1:{}, ...}
    ### YOUR CODE HERE ###

    # Convert image to hsv color map
    hsv = skimage.color.rgb2hsv(img[:, :, :3].astype('uint8'))

    # print(np.max(hsv[:,:,0]), np.max(hsv[:,:,1]), np.max(hsv[:,:,2]))
    # print(np.min(hsv[:,:,0]), np.min(hsv[:,:,1]), np.min(hsv[:,:,2]))

    # Count pixel positions
    for index_y, item in enumerate(img):

        for index_x, (r, g, b, l) in enumerate(item):

            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box of region l
            if R[l]["min_x"] > index_x:
                R[l]["min_x"] = index_x
            if R[l]["min_y"] > index_y:
                R[l]["min_y"] = index_y
            if R[l]["max_x"] < index_x:
                R[l]["max_x"] = index_x
            if R[l]["max_y"] < index_y:
                R[l]["max_y"] = index_y

    # Calculate the texture gradient
    gradient = calc_texture_gradient(img[:, :, :3])

    for l_key, value in R.items():

        # Calculate colour histogram and store in R
        masked_pixels = hsv[img[:, :, 3] == l_key]  # 2d regional HSV
        R[l_key]["size"] = len(masked_pixels)
        R[l_key]["hist_c"] = calc_colour_hist(masked_pixels)   # (75,)

        # Calculate texture histogram and store in R
        R[l_key]["hist_t"] = calc_texture_hist(gradient[img[:, :, 3] == l_key])

    return R


def extract_neighbours(regions):  # datastructure R, region dict

    def intersect(a, b):  # whether 4 corners of b in a
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###

    r_list = list(regions.items())
    for l, r_l in enumerate(r_list[:-1]):
        for r_rest in r_list[l + 1:]:
            if intersect(r_l[1], r_rest[1]):
                neighbours.append((r_l, r_rest))

    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE

    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])
    rt["size"] = new_size
    rt["hist_c"] = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size
    rt["hist_t"] = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size
    rt["labels"] = r1["labels"] + r2["labels"]

    return rt



def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)  # get 4 channel image

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]  # sort by S[i][1] ascending

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###

        key_to_delete = []
        for tuple, value in S.items():
            if (i in tuple) or (j in tuple):
                key_to_delete.append(tuple)

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###

        for k in key_to_delete:
            del S[k]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###

        key_to_delete.remove((i, j))
        for k in key_to_delete:
            n = k[1] if k[0] in (i, j) else k[0]  # neighbour of i or j
            S[(t, n)] = calc_sim(R[t], R[n], imsize)

    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###

    for k, v in R.items():
        regions.append({
            "rect": (v["min_x"], v["min_y"], v["max_x"] - v["min_x"], v["max_y"] - v["min_y"]),
            "labels": v["labels"],
            "size": v["size"]
        })

    return image, regions



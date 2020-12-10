from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats
from scipy.spatial import distance
from skimage import color
from skimage import io
import re
import math
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage import filters
from skimage.transform import hough_circle
from skimage.draw import circle_perimeter
from skimage.graph import route_through_array

def circlefit( image_filename      =   '',
               override_default    =   True,
               center              =   [None, None],
               crop                =   None,
               threshold           =   None,
               radii               =   [None, None] ):

    image = io.imread(image_filename)
    gray = color.rgb2gray(image)*float((2**16)-1)
    x_res, y_res = list(np.shape(gray))
    xc, yc = (x_res-1)/2, (y_res-1)/2

    plt.close()
    f = plt.figure(1, figsize=(6,4))
    ax1 = plt.subplot(2,3,1)

    # grayscale image
    ax1.imshow(gray, cmap='gray')
    # ax1.axvline(x = xc, linestyle='-', color='black')
    # ax1.axhline(y = yc, linestyle='-', color='black')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Grayscale Image')
    ax1.set_xlim(0,x_res)
    ax1.set_ylim(0,y_res)

    plt.show(block=False)

    # centered image
    if override_default == True or np.array_equal(center, [None, None]):
        center = np.array([int(xc), int(yc)])

    radius = int(min(center[0], center[1], x_res-1-center[0], y_res-1-center[1]))
    gray_centered = gray[center[1]-radius:center[1]+radius+1, center[0]-radius:center[0]+radius+1]

    ax2 = plt.subplot(2,3,2)
    ax2.set_title('Centered Image')
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(gray_centered, cmap='gray', extent=[-radius,+radius,-radius,+radius])
    ax2.set_xlim(-radius,+radius)
    ax2.set_ylim(-radius,+radius)
    # ax2.axvline(x = 0, linestyle='-', color='black')
    # ax2.axhline(y = 0, linestyle='-', color='black')
    plt.show(block=False)

    # cropped image
    if override_default == True or crop == None:
        crop = int(round(min(center[0], center[1], x_res-1-center[0], y_res-1-center[1]))) - 1

    gray_crop = gray_centered[radius-crop:radius+crop+1,radius-crop:radius+crop+1]

    ax3 = plt.subplot(2,3,3)
    ax3.set_title('Cropped Image')
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(gray_crop, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    plt.show(block=False)

    # edge sobel image
    gray_filter = filters.gaussian(gray_crop)
    edge_sobel = sobel(gray_filter).astype('int')
    min_val, max_val = edge_sobel.min(), edge_sobel.max()

    ax4 = plt.subplot(2,3,4)
    ax4.set_title('Edge Sobel Image')
    ax4.set_aspect('equal')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(edge_sobel, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])

    # binary image
    if override_default == True or threshold == None:
        binary = edge_sobel < int(threshold_otsu(edge_sobel))
    else:
        if threshold > 0:
            binary = edge_sobel > abs(threshold)
        else:
            binary = edge_sobel < abs(threshold)

    ax5 = plt.subplot(2,3,5)
    ax5.set_title('Binary Image')
    ax5.set_aspect('equal')
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    ax5.set_xlim(-crop, +crop)
    ax5.set_ylim(-crop, +crop)
    plt.show(block=False)

    if override_default == True or np.array_equal(radii, [None, None]):
        radii = [1, 2]

    hough_radii = np.arange(radii[0], radii[1], 1)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    x_circle_center = c
    y_circle_center = r
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])
    x_circle_perimeter = cc
    y_circle_perimeter = rr

    ax5.scatter(x_circle_center-crop, -(y_circle_center-crop), marker='x', color='red')
    ax5.scatter(x_circle_perimeter-crop, -(y_circle_perimeter-crop), marker='.', color='red')
    ax5.set_xlim(-crop, +crop)
    ax5.set_ylim(-crop, +crop)

    delta_x = center[0] - crop
    delta_y = center[1] - crop

    ax6 = plt.subplot(2,3,6)
    ax6.set_title('Circlefit Image')
    ax6.set_aspect('equal')
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.imshow(gray, cmap='gray')
    ax6.scatter(x_circle_center+delta_x, y_circle_center+delta_y, marker='x', color='black')
    ax6.scatter(x_circle_perimeter+delta_x, y_circle_perimeter+delta_y, marker='.', color='black')
    ax6.set_xlim(0,x_res)
    ax6.set_ylim(y_res,0)

    center_px = [x_circle_center+delta_x, y_circle_center+delta_y]
    radius_px = hough_radii[ridx]
    radius_max_px = int(round(min(center_px[0], center_px[1], x_res-1-center_px[0], y_res-1-center_px[1])))

    plt.show()

    return center_px, radius_px, radius_max_px

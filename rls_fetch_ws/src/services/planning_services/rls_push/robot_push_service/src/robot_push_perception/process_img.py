#!/usr/bin/env python

import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import cv2
import numpy as np


W = 124
H = 106
#W = 128
#H = 106
STEP = 5  # degree


def center_img(img):
    center = get_center(img)
    ori_img = transform_img(img.copy(), 0, int(
        W/2 - center[1]), int(H/2 - center[0]))
    return ori_img, center - np.array([H/2, W/2])


def get_center(img):
    '''
    return the geometric center of an image blob
    '''
    if len(img.shape) == 3:
        img = img.copy()[:, :, 0]

    (yidx, xidx) = np.where(img > 1)
    coords = np.array([[yidx[i], xidx[i]] for i in range(len(yidx))])
    center = coords.mean(axis=0)

    return center


def get_rot_rect(img):
    '''only for the purpose of pushing object to the side while aligning the shorter side with table edge'''
    cnts, _ = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cnts[0]
    rect = cv2.minAreaRect(c)
    w = rect[1][0]
    h = rect[1][1]
    degree = rect[2]

    if w < h:
        degree += 90.0

    angle = degree
    if degree < 0:
        degree = -degree
    else:
        degree = 180 - degree

    return degree, min(w, h), max(w, h), angle


def get_goal_w(img):
    '''only for the purpose of pushing object to the side while aligning the shorter side with table edge'''
    cnts, _ = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cnts[0]
    rect = cv2.minAreaRect(c)
    w = rect[1][0]
    h = rect[1][1]
    degree = rect[2]

    if w < h:
        degree += 90.0

    return degree


def get_img_transform(from_img, to_img, symmetric=False, prev_diff=0):
    '''
    get rotation and translation required to move from_img to to_img
    '''
    from_center = get_center(from_img)
    to_center = get_center(to_img)
    # get diff in translation
    diff_tran = to_center - from_center

    # move both to center of the image
    from_img_ori = transform_img(from_img.copy(), 0, int(
        W/2 - from_center[1]), int(H/2 - from_center[0]))
    to_img_ori = transform_img(to_img.copy(), 0, int(
        W/2 - to_center[1]), int(H/2 - to_center[0]))

    # rotate from_img in 360 degree to check how much it overlaps with to_img
    max_overlap = -10000
    best_w = 0
    best_w_list = []
    for i in range(int(180/STEP)):
        dw = -90 + i * STEP
        dummy_img = transform_img(from_img_ori.copy(), dw, 0, 0)
        num_overlap = count_overlap(dummy_img.copy(), to_img_ori.copy())
        if num_overlap > max_overlap:
            max_overlap = num_overlap
            best_w = dw
            if num_overlap > 0.95:
                best_w_list.append(dw)

    # if 90 < best_w < 180:
    #    if symmetric:
    #        best_w -= 180
    #    else:
    #        best_w -= 360

    # for i, b in enumerate(best_w_list):
    #    if b > 180:
    #        if symmetric:
    #            best_w_list[i] -= 180
    #        else:
    #            best_w_list[i] -= 360

    if max_overlap > 0.95:
        idx = np.argmin(np.abs(np.array(best_w_list) - prev_diff))
        return diff_tran, best_w_list[idx]
    else:
        return diff_tran, best_w


def generate_goal_img(img, w, x, y):
    ''' generate goal image in original image frame'''
    center = get_center(img)
    img_ori = transform_img(img.copy(), 0, int(
        W/2 - center[1]), int(H/2 - center[0]))
    img_ = transform_img(img_ori.copy(), w, x, y)
    img_f = transform_img(
        img_.copy(), 0, -int(W/2 - center[1]), -int(H/2 - center[0]))

    return img_f


def generate_goal_img_real(img, dw, dx, dy):
    ''' generate goal image in original image frame'''
    w = 640
    h = 480
    center = get_center(img)
    img_ori = transform_img(img.copy(), 0, int(
        w/2 - center[1]), int(h/2 - center[0]))
    img_ = transform_img(img_ori.copy(), dw, dx, dy)
    img_f = transform_img(
        img_.copy(), 0, -int(w/2 - center[1]), -int(h/2 - center[0]))

    return img_f


def generate_img_pair(img, dw, dx, dy):
    '''generate realigned curr and next images'''
    center = get_center(img)
    img_ori = transform_img(img.copy(), 0, int(
        W/2 - center[1]), int(H/2 - center[0]))
    img_g = transform_img(img_ori.copy(), dw, dx, dy)

    return img_ori, img_g


def transform_img(img, w, x, y):
    '''
    rotate by w (degree) first
    then translate by x, y (pixel)
    '''
    img_rot = imutils.rotate(img.copy(), w)
    img_tran = imutils.translate(img_rot.copy(), x, y)
    return img_tran


def count_overlap(img1, img2):
    '''
    count number of overlapping pixels
    '''
    (y1, x1) = np.where(img1 > 1)
    (y2, x2) = np.where(img2 > 1)

    set1 = set(((y1[i], x1[i]) for i in range(len(y1))))
    set2 = set(((y2[i], x2[i]) for i in range(len(y2))))

    return 1.0 * len(set1.intersection(set2)) / len(y2)


'''following functions measure the pixel to metric scale'''


def get_pixel_metric(img, l, w):
    '''Input: real image, longer side l in real metric, shorter side w in real
    metric'''
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts, method='right-to-left')
    pixelsPerMetric = None
    center = None

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        rect = cv2.minAreaRect(c)
        (ww, hh) = rect[1]

        box = cv2.cv.BoxPoints(rect)
        box = np.array(box, dtype='int')
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        if pixelsPerMetric is not None:
            print(pixelsPerMetric)
            print(max(ww, hh) / pixelsPerMetric)
        else:
            pixelsPerMetric = max(ww, hh) / max(l, w)
            center = rect[0]  # (x, y)
            #print pixelsPerMetric
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # break

    return pixelsPerMetric, center


def generate_input_image(points):
    """
    generate Push-Net input image given a set of points representing the segmented object
    """
    # find positional center
    (xc, yc) = get_point_center(points)

    # map to an image
    img_mapped = point_to_image(points, xc, yc)

    # find contour and fill up the region within contour
    img_filled = fill_contour(img_mapped)

    return img_filled


def get_point_center(points):
    x_avg = 0
    y_avg = 0
    num = len(points)
    for p in points:
        x_avg += p.x
        y_avg += p.y
    return x_avg / num, y_avg / num


############### Test Functions ##############################################
#############################################################################
def test_scale():
    img = cv2.imread('color.jpg')
    p2m = get_pixel_metric(img, 0.09, 0.05)  # dimension of nus tmc namecard

# test_scale()


def test_transform():
    img = cv2.imread('test.jpg')[:, :, 0]
    w = 19
    x = 42
    y = 5
    center = get_center(img.copy())
    img_t = transform_img(img.copy(), 0, int(
        W/2-center[1]), int(H/2-center[0]))

    img_ = transform_img(img_t.copy(), w, x, y)
    img_f = transform_img(
        img_.copy(), 0, -int(W/2-center[1]), -int(H/2-center[0]))

    #cv2.imshow('img', img)
    # cv2.waitKey(0)
    #cv2.imshow('img', img_t)
    # cv2.waitKey(0)
    #cv2.imshow('img', img_)
    # cv2.waitKey(0)
    #cv2.imshow('img', img_f)
    # cv2.waitKey(0)

    print(get_img_transform(img.copy(), img_f.copy()))

# test_transform()

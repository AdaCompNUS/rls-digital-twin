import cv2
import numpy as np
from push_vision.process_img import *


W = 128.0
H = 106.0

def sample_action_new(img, num_actions):
    s = 0.8
    safe_margin = 1.4
    out_margin = 2.0

    img_inner = cv2.resize(img.copy(), (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = img_inner.shape
    img_end = np.zeros((int(H), int(W)))
    img_end[(int(H)-h)/2:(int(H)+h)/2, (int(W)-w)/2:(int(W)+w)/2] = img_inner.copy()
    (inside_y, inside_x) = np.where(img_end.copy()>0)

    ## sample start push point outside a safe margin of object
    img_outer1 = cv2.resize(img.copy(), (0,0), fx=safe_margin, fy=safe_margin, interpolation=cv2.INTER_CUBIC)
    h, w = img_outer1.shape
    img_start_safe = np.zeros((int(H), int(W)))
    img_start_safe = img_outer1.copy()[(h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

    img_outer2 = cv2.resize(img.copy(), (0,0), fx=out_margin, fy=out_margin, interpolation=cv2.INTER_CUBIC)
    h, w = img_outer2.shape
    img_start_out = np.zeros((int(H), int(W)))
    img_start_out = img_outer2.copy()[(h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

    img_start = img_start_out.copy() - img_start_safe.copy()
    (outside_y, outside_x) = np.where(img_start.copy()>0)


    #cv2.imshow('ori', img)
    #cv2.waitKey(0)
    #cv2.imshow('small', img_end)
    #cv2.waitKey(0)
    #cv2.imshow('safe', img_start_safe)
    #cv2.waitKey(0)
    #cv2.imshow('out', img_start_out)
    #cv2.waitKey(0)
    #cv2.imshow('start', img_start)
    #cv2.waitKey(0)

    num_inside = len(inside_x)
    num_outside = len(outside_x)

    actions = []
    for i in range(num_actions):
        ## sample an inside point
        inside_idx = np.random.choice(num_inside)
        outside_idx = np.random.choice(num_outside)
        ## sample an outside point
        start_x = 0
        start_y = 0
        end_x = 0
        end_y = 0
        while True:
            end_x = int(inside_x[inside_idx])
            end_y = int(inside_y[inside_idx])
            start_x = int(outside_x[outside_idx])
            start_y = int(outside_y[outside_idx])

            if start_x < 0 or start_x >= W or start_y < 0 or start_y >= H:
                print 'out of bound'
                continue
            if img[start_y, start_x] == 0:
                break
            else:
                #print img[start_y, start_x]
                continue

        actions.append(start_x)
        actions.append(start_y)
        actions.append(end_x)
        actions.append(end_y)
    return actions

def draw_action(img, start, end, offset=[0,0], last=False):
    (yy, xx) = np.where(img>0)
    img_3d = np.zeros((int(H), int(W), 3))
    img_3d[yy, xx] = np.array([255,255,255])
    #print img_3d.shape
    sx = int(start[0] + offset[1])
    sy = int(start[1] + offset[0])
    ex = int(end[0] + offset[1])
    ey = int(end[1] + offset[0])
    #print sx, sy, ex, ey
    cv2.line(img_3d, (sx, sy), (ex, ey), (0,0,255), 3)
    #self.save_img(img_3d, 'mask_'+str(self.curr_step))
    #img_3d = img_3d.astype(int)

    cv2.imshow('action', img_3d)
    if last:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)


img = cv2.imread('test.jpg')[:,:,0]
print img.shape
num_action = 1000
img, center_offset = center_img(img.copy())
h, w = img.shape
for i in range(h):
    for j in range(w):
        if img[i, j] >0:
            print img[i, j]
_, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
actions = sample_action_new(img.copy(), num_action)
for i in range(num_action):
    best_start = actions[4*i:4*i+2]
    best_end = actions[4*i+2:4*i+4]
    draw_action(img.copy(), best_start, best_end, last=False)


import argparse
import cv2
import os, sys
import numpy as np
import datetime
import imutils
import math


top_cascade = cv2.CascadeClassifier('HAAR.xml')
SZ_LIMIT1 = 120
SZ_LIMIT2 = 200

ROI_HEIGHT = 20

# check if (x,y) is in d-neighbour of (x0,y0)
def testNeighbourIn(x, y, x0, y0, d):
    #d = 30
    rc = ((x0-d, y0-d), (x0+d, y0+d))
    if x > x0-d and x < x0+d and y > y0-d and y < y0+d:
        return True
    return False

def checkFixed(prvs, curs):
    result = []
    if len(curs) == 0:
        return result
    for box in prvs:
        [cx, cy, w] = box
        isAgain = False
        for newbox in curs:
            [cx1, cy1, w1] = newbox
            if testNeighbourIn(cx, cy, cx1, cy1, 25) or not testNeighbourIn(cx, cy, cx1, cy1, 40):
                isAgain = True
                break
        if not isAgain:
            result.append(box)
    return result

def countFromFile(inVideo, outVideo):
    if os.path.exists(outVideo):
        os.remove(outVideo)
    total_passenger = 0
    cap = cv2.VideoCapture(inVideo)

    frame_width = 640
    frame_height = 480
    ini_frame = 0
    prev_frame = 0

    if not cap.isOpened():
        print("Can't read video file: {}".format(inVideo))
    else:
        ret, img = cap.read()
        frame_height = len(img)
        frame_width = len(img[0])
    first_cap = True
    prev_boxes = []
    cur_boxes = []
    # out = cv2.VideoWriter(outVideo, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    msg = ""
    nFrames = 0
    [ROI_X, ROI_Y] = [10, frame_height//3]
    [ROI_W, ROI_H] = [frame_width-20, frame_height-ROI_Y-10]

    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        nFrames += 1
        if nFrames < 250:
            ini_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            prev_frame = ini_frame
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        peoples = top_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (SZ_LIMIT1, SZ_LIMIT1), (SZ_LIMIT2, SZ_LIMIT2))

        # draw ROI rectangle
        cv2.line(img, (ROI_X, ROI_Y+ROI_H//2), (ROI_X+ROI_W, ROI_Y+ROI_H//2), (255, 0, 255), 2, 1)
        cv2.rectangle(img, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 0, 255), 2)

        cur_boxes = []
        # draw detected rect
        for (x, y, w, h) in peoples:
            thresh1 = 0
            thresh2 = 0
            try:
                frameDelta = cv2.absdiff(prev_frame[x:x+w-1, y:y+h-1], gray[x:x+w-1, y:y+h-1])
                thresh1 = np.mean(frameDelta)
                frameDelta = cv2.absdiff(ini_frame[x:x + w - 1, y:y + h - 1], gray[x:x + w - 1, y:y + h - 1])
                thresh2 = np.mean(frameDelta)
            except:
                continue
            if thresh1 < 15: continue
            if thresh2 < 15: continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # calculate center point
            [cx, cy] = [x + w//2, y + h//2]
            cv2.circle(img, (cx,cy), 2, (0,0,255), 3)
            #'''
            if cx > ROI_X and cx < ROI_X+ROI_W and cy > ROI_Y and cy < ROI_Y+ROI_H//2:
                cur_boxes.append((cx,cy,w))
            #'''

        if not first_cap:
            prev_boxes = checkFixed(prev_boxes, cur_boxes)
            total_passenger += len(prev_boxes)

        prev_boxes = cur_boxes

        if first_cap:
            first_cap = False
            total_passenger = 0
        msg = "total passenger: {}".format(total_passenger)
        cv2.putText(img, msg, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255))

        cv2.imshow('result', img)
        # out.write(img)

        prev_frame = gray
        flag = True
        while 1:
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                flag = False
                break
            elif k == ord('c'):
                break
        if not flag:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

    print(msg)

def countFromLiveCam(outVideo):
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(outVideo, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while (True):
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        peoples = top_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (SZ_LIMIT1, SZ_LIMIT1),
                                               (SZ_LIMIT2, SZ_LIMIT2))

        for (x, y, w, h) in peoples:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('result', img)
        out.write(img)
        k = cv2.waitKey(33) & 0xff
        if k == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=str, default="yes")
    parser.add_argument("--video", type=str, default="video/test.avi")
    parser.add_argument("--out", type=str, default="video/out.avi")

    args = parser.parse_args()

    if args.live == "no":
        countFromFile(args.video, args.out)
    else:
        countFromLiveCam(args.out)
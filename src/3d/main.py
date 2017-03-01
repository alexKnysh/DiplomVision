# -*-coding: utf-8-*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def saveImg(n, img_l, img_r):
    cv2.imwrite("test/left_" + str(n) + ".png", img_l)
    cv2.imwrite("test/right_" + str(n) + ".png", img_r)
    n = n + 1
    return n;


def ReadCam():
    cam = cv2.VideoCapture(0)  # левая
    cam0 = cv2.VideoCapture(2)  # правая
    n = 0;
    while (1):
        ret_val, frame = cam.read()
        ret_val0, frame0 = cam0.read()
        cv2.imshow('frame', frame)
        cv2.imshow('frame0', frame0)
        k = cv2.waitKey(30) & 0xff
        # рендринг материала для калибровки
        if k == 10:
            n = saveImg(n, frame, frame0)
        if k == 27:
            break


def calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    imagePoints1 = []  # 2d points in image plane.
    imagePoints2 = []  # 2d points in image plane.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # чтение уже в градациях серого
    imgl = cv2.imread('calibration/left/0.png')
    imgr = cv2.imread('calibration/right/0.png')
    img_l = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
    img_r = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    height, width = img_l.shape
    ret_l, coerce_l = cv2.findChessboardCorners(img_l, (9, 6), None)
    ret_r, coerce_r = cv2.findChessboardCorners(img_r, (9, 6), None)

    if ret_l and ret_r:
        cv2.cornerSubPix(img_l, coerce_l, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(img_r, coerce_r, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(imgl, (9, 6), coerce_l, ret_l)
        cv2.drawChessboardCorners(imgr, (9, 6), coerce_r, ret_r)
        # cv2.imshow('img_l_find', imgl)
        # cv2.imshow('img_r_find', imgr)
        # cv2.waitKey()

    if ret_l and ret_r:
        objpoints.append(objp)
        imagePoints1.append(coerce_l)
        imagePoints2.append(coerce_r)
        try:
            retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                                                             imagePoints1,
                                                                                                             imagePoints2,
                                                                                                             (width,
                                                                                                              height))
            print 'R = ', R, '\n'
            print 'T = ', T, '\n'
            print 'E = ', E, '\n'
            print 'F = ', F, '\n'

            R1 = np.zeros(shape=(3, 3))
            R2 = np.zeros(shape=(3, 3))
            P1 = np.zeros(shape=(3, 4))
            P2 = np.zeros(shape=(3, 4))

            cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (width, height), R, T, R1, R2, P1,
                              P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0))



            map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height),
                                                       cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height),
                                                       cv2.CV_32FC1)

        except ValueError:
            print ValueError

        return (map1x, map1y, map2x, map2y)
        # imgU1 = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
        # imgU2 = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)
        # cv2.imshow('right', img_r)
        # cv2.imshow('left', img_l)
        # cv2.imshow('right_U', imgU1)
        # cv2.imshow('left_U', imgU2)
        # stereo = cv2.StereoBM()
        # disparity = stereo.compute(img_l,img_r,disparity=cv2.CV_32F, disptype=cv2.CV_32F)
        # stereo = cv2.StereoBM(1, 32, 15)
        # disparity = stereo.compute(imgU2, imgU1)
        # plt.imshow(disparity, 'gray')
        # cv2.waitKey()
        # plt.show()



def main():
    (map1x, map1y, map2x, map2y) = calibration()

    cam = cv2.VideoCapture(0)  # левая
    cam0 = cv2.VideoCapture(2)  # правая
    n = 0;
    while (1):
        ret_val, frame = cam.read()
        ret_val0, frame0 = cam0.read()
        img_l = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        height, width, depth = frame.shape
        imgU1 = np.zeros((height, width, 3), np.uint8)
        imgU2 = np.zeros((height, width, 3), np.uint8)
        imgU1 = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
        imgU2 = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)

        stereo = cv2.StereoBM(1,0,11)
        disparity = stereo.compute(imgU2, imgU1, cv2.CV_32F)
        # cv2.imshow('disparity', disparity)
        cv2.imshow('right_U', imgU1)
        cv2.imshow('left_U', imgU2)
        cv2.imshow('frame', frame)
        cv2.imshow('frame0', frame0)

        plt.imshow(disparity, 'gray')
        plt.show()
        k = cv2.waitKey(30) & 0xff

        # рендринг материала для калибровки
        if k == 10:
            n = saveImg(n, frame, frame0)
        if k == 27:
            break
            # ReadCam()


if __name__ == '__main__':
    print ('----start-----')
    main()
    print ('-----end-----')

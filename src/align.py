""" Выравнивание (поворот, сдвиг) """

# coding: utf-8

import numpy as np
import cv2

def rotate_image(img,angle):
    rows,cols = img.shape[:2]
    M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst=cv2.warpAffine(img,M,(cols,rows))

    return dst, M


def get_rotation_angle(imgray):

    thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle

    return angle


def main(img):
    ''' img is not gray scale '''
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgray = cv2.bitwise_not(imgray)

    angle = get_rotation_angle(imgray)

    (img_rot,_)= rotate_image(img, angle)
    return img_rot

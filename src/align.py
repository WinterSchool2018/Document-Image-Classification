""" Выравнивание (поворот, сдвиг) """

# coding: utf-8

# In[1]:


import numpy as np
import cv2
import skimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import color
from skimage import io

img = cv2.imread('image.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray = cv2.bitwise_not(imgray)

def rotate_image(img,angle):
    rows,cols = img.shape[:2]
    M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst=cv2.warpAffine(img,M,(cols,rows))

    return dst, M


# In[ ]:

def get_rotation_angle(imgray):

    thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle

    return angle

angle = get_rotation_angle(imgray)

(img_rot,_)= rotate_image(img, angle)

plt.imshow(img_rot)
plt.show()

cv2.imwrite('image1.png',img_rot)


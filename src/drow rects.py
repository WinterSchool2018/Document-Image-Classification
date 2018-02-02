
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#fields on the first page
rects1 = [((1497, 231), (1505,254)),
((1567, 238), (1604, 252)),
((1586, 423), (2355, 452)),
((568, 598), (582, 612)),
((998, 598), (1012, 615)),
((1046, 601), (1061, 618)),
((1146, 604), (1158, 619)),
((1898,314), (1912, 332))]

#fields on the second page
rects2 = [((672, 1694),(692, 1708)),
((748, 1696), (772, 1709)),
((1725, 2281), (2394,2297)),
((1393, 2320), (1954, 2333)),
((1799, 2361), (2203, 2377)),
((1179, 2773), (1711, 2826))]


# In[3]:


filename = 'dog2_empty.jpg'
rects = rects2
document = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#cv2.imshow('test',document)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[ ]:


def drow_rectangle(img, p1, p2):
    cv2.rectangle(img, p1, p2, (0, 0, 255), 1)


# In[5]:


def calc_image(img):
    """
    Функция возвращает процент чернил в картинке
    """
    thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    count_of_pixel = np.count_nonzero(thresholded)
    percent = 100 * np.count_nonzero(thresholded) / (img.size)

    return percent

def is_signature(img, threshold=10):
    """
    Проверяем есть ли подпись в ``img``
    :param threshold: порог, при котором мы считаем, что подпись есть
    :param img: вырезанное изображение подписи
    :return: true/false
    """
    percent = calc_image(img)
    return percent > threshold, round(percent, 3)


# In[6]:


def drow_rects(document, rects):
    for rect in rects:
        p1,p2 = rect
        #x1,y1 = p1
        #x2,y2 = p2
        drow_rectangle(document, p1, p2)
        #cv2.imshow('test', field_img)
     
    #plt.imshow(document, cmap='gray')
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    cv2.imshow('test',document)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# In[ ]:


drow_rects(document,rects)


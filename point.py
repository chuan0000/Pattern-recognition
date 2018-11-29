##資料整理，標記出鴨子
##日期2018/11/28
##作者徐信銓
import numpy as np
import math
import cv2
from PIL import Image
import os
import sys
import glob as gb


def _sun(image):

    w,h = image.shape
    image=np.reshape(image,(1,w*h))[0]

    sun = image.sum()
    return sun//255
image = cv2.imread('C:/000/Pattern-recognition/fullbaseduckimage9.tif')

b,g,r=cv2.split(image)
w,h=b.shape

##binaryimage 
b=0
g=0
r=0
c=0
im=Image.new('L',(h,w))
for y in range(h):
    for x in range(w):
        b=int(image[0][x,y])
        g=int(image[1][x,y])
        r=int(image[2][x,y])
        if b==0 and g==0 and r==255:
            c = 255
        else:
            c=0
        im.putpixel((y,x),c)
im.save('binary.tif')

binary = cv2.imread('binary.tif',0)
sun = _sun(binary)
print(sun)
image = cv2.imread('C:/000/Pattern-recognition/image/full_duck.jpg')
mage3 ,contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
CENTER =np.zeros((100000,1,2),int)
j=0
for cnt in contours:
    area = int(cv2.contourArea(cnt))
    if area>=200 :
        if int(len(cnt[:,0])) >0:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            CENTER[j][0] = center
    j=j+1
print('len(CENTER)',len(CENTER))
_CENTER=[[]]
for _len in range(len(CENTER)):
    if CENTER[_len].any() !=0:
        _CENTER[0].append(CENTER[_len][0])
_CENTER = np.array(_CENTER,int)
print('len(_CENTER[0])',len(_CENTER[0]))
for _len in range(len(_CENTER[0])):
    xx = _CENTER[0][_len][0]
    yy = _CENTER[0][_len][1]

    img = image[yy-25:yy+25,xx-25:xx+25]
    if sun>200:
        image[yy-5:yy+5,xx-5:xx+5] = (255,0,0)
    else:
        image[yy-25:yy+25,xx-25:xx+25] = img
        
cv2.imwrite('newimage.tif',image)


             

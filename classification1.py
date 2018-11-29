##計算g(x)
##日期2018/11/28
##作者徐信銓
##label1:鴨子(R,G,B之Pixel大於220)
##label2:河道
##label3:石頭

import numpy as np
import math
import cv2
from PIL import Image
import os
import sys
import glob as gb
t1 = cv2.getTickCount()
##################################################################
#計算mu
def _mu(duckimage):
    b_duckimage,g_duckimage,r_duckimage = cv2.split(duckimage)
    w_duckimage,h_duckimage = b_duckimage.shape
    b_pixel=0
    g_pixel=0
    r_pixel=0
    b_254=0
    g_254=0
    r_254=0
    ##灰階圖算非零點的個數
    for y in range(h_duckimage):
        for x in range(w_duckimage):
            b_pix = int(b_duckimage[x,y])
            g_pix = int(g_duckimage[x,y])
            r_pix = int(r_duckimage[x,y])
            if b_pix >0 :
                b_pixel=b_pixel+b_pix
                b_254 = b_254+1
            else:
                b_pixel=b_pixel
                b_254 = b_254
            if g_pix >0 :
                g_pixel=g_pixel+g_pix
                g_254 = g_254+1
            else:
                g_pixel=g_pixel
                g_254 = g_254
            if r_pix >0 :
                r_pixel=r_pixel+r_pix
                r_254 = r_254+1
            else:
                r_pixel=r_pixel
                r_254 = r_254
    return b_pixel,g_pixel,r_pixel,b_254,g_254,r_254

############################################################
##計算sigma
def _sigma(duckimage,mu1,mu2,mu3):
    b_duckimage,g_duckimage,r_duckimage = cv2.split(duckimage)
    w_duckimage,h_duckimage = b_duckimage.shape
    b_pixel=0
    g_pixel=0
    r_pixel=0
    g_255=0
    b_255=0
    r_255=0
    _SIGMA=[]
    for y in range(h_duckimage):
        for x in range(w_duckimage):
            b_pix = int(b_duckimage[x,y])
            g_pix = int(g_duckimage[x,y])
            r_pix = int(r_duckimage[x,y])
            if b_pix >0 :
                b_pixel=b_pixel+(b_pix-mu1)**2
                b_255=b_255+1
            else:
                b_pixel=b_pixel
                b_255 = b_255
            if g_pix >0 :
                g_pixel=g_pixel+(g_pix-mu2)**2
                g_255=g_255+1
            else:
                g_pixel=g_pixel
                g_255=g_255
            if r_pix >0 :
                r_pixel=r_pixel+(r_pix-mu3)**2
                r_255=r_255+1
            else:
                r_pixel=r_pixel
                r_255=r_255
    return b_pixel,g_pixel,r_pixel,b_255,g_255,r_255

##########################################################
##計算p(x)
def _probability(matrix,d,mu1,mu2,mu3,det,inv):
    aa=0
    bb=0
    cc=0
    dd=0
    ee=0
    ff=0
    gg=0
    pi = int(math.pi*100)/100
    aa=(2*pi)**(d/2)
    bb=det**(1/2)
    cc = [[matrix[0]-mu1,matrix[1]-mu2,matrix[2]-mu3]]
    cc =np.array(cc[0])
    dd = inv
    ddd = np.array([cc[0]*dd[0][0],cc[1]*dd[1][1],cc[2]*dd[2][2]])
    ee = [[matrix[0]-mu1],[matrix[1]-mu2],[matrix[2]-mu3]]
    ee = np.array(ee)
    eee = ddd[0]*ee[0][0]+ddd[1]*ee[1][0]+ddd[2]*ee[2][0] 
    ff = np.exp((-1/2)*eee)
    gg = 1/(aa*bb)*ff
    return gg
########################################################
##label 1
datapath_1 = gb.glob('C:/000/Pattern-recognition/label1_sample/*.tif')
mu1=0
mu2=0
mu3=0
b_24=0
g_24=0
r_24=0
sig1=0
sig2=0
sig3=0
b_25=0
g_25=0
r_25=0
for i  in datapath_1:
    image = cv2.imread(i)
    mu =_mu(image)
    mu1=mu1+mu[0]
    b_24=b_24+mu[3]
    mu2=mu2+mu[1]
    g_24=g_24+mu[4]
    mu3=mu3+mu[2]
    r_24=r_24+mu[5]
mu1_b = int(float(mu1/b_24)*1000)/1000
mu1_g = int(float(mu2/g_24)*1000)/1000
mu1_r = int(float(mu3/r_24)*1000)/1000
mulabel1 = [[mu1_b],[mu1_g],[mu1_r]]
print('mu1',mulabel1)
for ii  in datapath_1:
    image = cv2.imread(ii)
    b_sigma,g_sigma,r_sigma,b255,g255,r255 = _sigma(image,mu1_b,mu1_g,mu1_r)
    sig1=sig1+b_sigma
    b_25=b_25+b255
    sig2=sig2+g_sigma
    g_25=g_25+g255
    sig3=sig3+r_sigma
    r_25=r_25+r255
_SIGMA1 = np.array([[int(float(sig1/b_25)*100)/100,0,0],[0,int(float(sig2/g_25)*100)/100,0],[0,0,int(float(sig3/r_25)*100)/100]])
print('SIGMA1',_SIGMA1)
_det1 = np.linalg.det(_SIGMA1)
print('det1',_det1)
_inv1 = np.linalg.inv(_SIGMA1)
print('inv_SIGMA1',_inv1)
######################################################################
##label 2
datapath_2 = gb.glob('C:/000/Pattern-recognition/label2_sample/*.tif')
mu1=0
mu2=0
mu3=0
b_24=0
g_24=0
r_24=0
sig1=0
sig2=0
sig3=0
b_25=0
g_25=0
r_25=0
for iii  in datapath_2:
    image = cv2.imread(iii)
    b_pixel,g_pixel,r_pixel,b254,g254,r254 =_mu(image)
    print('b_pixel,g_pixel,r_pixel,b254,g254,r254',b_pixel,g_pixel,r_pixel,b254,g254,r254)
    mu1=mu1+b_pixel
    b_24=b_24+b254
    mu2=mu2+g_pixel
    g_24=g_24+g254
    mu3=mu3+r_pixel
    r_24=r_24+r254
mu2_b = int(float(mu1/b_24)*1000)/1000
mu2_g = int(float(mu2/g_24)*1000)/1000
mu2_r = int(float(mu3/r_24)*1000)/1000
mulabel2 = [[mu2_b],[mu2_g],[mu2_r]]
print('mu2',mulabel2)
for iiii  in datapath_2:
    image = cv2.imread(iiii)
    b_sigma,g_sigma,r_sigma,b255,g255,r255 = _sigma(image,mu2_b,mu2_g,mu2_r)
    sig1=sig1+b_sigma
    b_25=b_25+b255
    sig2=sig2+g_sigma
    g_25=g_25+g255
    sig3=sig3+r_sigma
    r_25=r_25+r255
_SIGMA2 = np.array([[int(float(sig1/b_25)*100)/100,0,0],[0,int(float(sig2/g_25)*100)/100,0],[0,0,int(float(sig3/r_25)*100)/100]])
print('SIGMA2',_SIGMA2)
_det2 = np.linalg.det(_SIGMA2)
print('det2',_det2)
_inv2 = np.linalg.inv(_SIGMA2)
print('inv_SIGMA2',_inv2)
################################################################
##label 3
datapath_3 = gb.glob('C:/000/Pattern-recognition/label3_sample/*.tif')
mu1=0
mu2=0
mu3=0
b_24=0
g_24=0
r_24=0
sig1=0
sig2=0
sig3=0
b_25=0
g_25=0
r_25=0
for iiiii  in datapath_3:
    image = cv2.imread(iiiii)
    b_pixel,g_pixel,r_pixel,b254,g254,r254 =_mu(image)
    print('b_pixel,g_pixel,r_pixel,b254,g254,r254',b_pixel,g_pixel,r_pixel,b254,g254,r254)
    mu1=mu1+b_pixel
    b_24=b_24+b254
    mu2=mu2+g_pixel
    g_24=g_24+g254
    mu3=mu3+r_pixel
    r_24=r_24+r254
mu3_b = int(float(mu1/b_24)*1000)/1000
mu3_g = int(float(mu2/g_24)*1000)/1000
mu3_r = int(float(mu3/r_24)*1000)/1000
mulabel3 = [[mu3_b],[mu3_g],[mu3_r]]
print('mu3',mulabel3)
for iiiiii  in datapath_3:
    image = cv2.imread(iiiiii)
    b_sigma,g_sigma,r_sigma,b255,g255,r255 = _sigma(image,mu3_b,mu3_g,mu3_r)
    sig1=sig1+b_sigma
    b_25=b_25+b255
    sig2=sig2+g_sigma
    g_25=g_25+g255
    sig3=sig3+r_sigma
    r_25=r_25+r255
_SIGMA3 = np.array([[int(float(sig1/b_25)*100)/100,0,0],[0,int(float(sig2/g_25)*100)/100,0],[0,0,int(float(sig3/r_25)*100)/100]])
print('SIGMA3',_SIGMA3)
_det3 = np.linalg.det(_SIGMA3)
print('det3',_det3)
_inv3 = np.linalg.inv(_SIGMA3)
print('inv_SIGMA3',_inv3)

##############################################################
print('start')
log1=0
log2=0
log3=0
loga=0
logb=0
logc=0
##imgpath = gb.glob('C:/work/classification1/image/*.tif')
imgpath = gb.glob('C:/000/Pattern-recognition/image/*.jpg')
z=1
for j in imgpath:
    print(j)
    image = cv2.imread(j)
    image = cv2.split(image)
    w,h = image[0].shape
    print(w,h)
    a= []
    log=0
    _log=[]
    c=0
    im = Image.new('RGB',(h,w))
    for y  in range(h):
        for x in range(w): 
            b = int(image[0][x,y])
            g = int(image[1][x,y])
            r = int(image[2][x,y])
            a = [b,g,r]
            probability1=_probability(a,3,mu1_b,mu1_g,mu1_r,_det1,_inv1)
            probability2=_probability(a,3,mu2_b,mu2_g,mu2_r,_det2,_inv2)
            probability3=_probability(a,3,mu3_b,mu3_g,mu3_r,_det3,_inv3)
            if probability1==0 and probability2 !=0 and probability3 !=0 :
                c = (r,g,b)
                im.putpixel((y,x),c)
            elif probability1==0 and probability2 !=0 and probability3 ==0 :
                c = (r,g,b)
                im.putpixel((y,x),c)
            elif probability1!=0 and probability2 ==0 and probability3 !=0 :

                log1 = int(float(math.log(probability1,10))*100)/100
                log3 = int(float(math.log(probability3,10))*100)/100
                if log1>log3:
##                    print('a:',log3,log1)
                    c =(255,0,0)
                elif log3>log1 and log1>-100 and log3<-20:
##                    print('b:',log3,log1)
                    c=(255,0,0)
                else:
##                    print('c:',log3,log1)
                    c = (r,g,b)
                im.putpixel((y,x),c)
            elif probability1!=0 and probability2 ==0 and probability3 ==0 :
                
                log1 = int(float(math.log(probability1,10))*100)/100

                if log1>-6:
                    c =(255,0,0)
                elif log1<-6:
                    c =(255,0,0)
                im.putpixel((y,x),c)
            else:
                c=(r,g,b)
                im.putpixel((y,x),c)
                    

                              
    im.save('fullbaseduckimage.tif')
    baseduckimage = cv2.imread('fullbaseduckimage.tif')
    cv2.imwrite('fullbaseduckimage%d.tif'%int(z),baseduckimage)
    
    z= z+1


t2 = cv2.getTickCount()

t_2 = (t2-t1)/cv2.getTickFrequency()
print ("excution time:",t_2)


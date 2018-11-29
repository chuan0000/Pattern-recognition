import cv2
import numpy as np
from PIL import Image
import glob as gb
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated

##C:/work/full_duck.jpg
image = cv2.imread('C:/work/full_duck.jpg')

##cv2.imshow('a',image)


##print(image)
image = cv2.split(image)
w,h = image[0].shape
print(w,h)

##image[0] = np.reshape(image[0],(1,w*h))
##image[1] = np.reshape(image[1],(1,w*h))
##image[2] = np.reshape(image[2],(1,w*h))
##print('0',list(image[0][0]))
##print('1',list(image[1][0]))
##print('2',list(image[2][0]))

c = 0
im = Image.new('RGB',(h,w))
for y in range(h):
    for x in range(w):
        b = int(image[0][x,y])
        g = int(image[1][x,y])
        r = int(image[2][x,y])
        if b>=220 and g>=220 and r>=220 :
            c = (r,g,b)
        else:
            c=(0,0,0)
        im.putpixel((y,x),c)
im.save('duck240.tif')

####################################################

##imag  = cv2.imread('C:/work/oneduck.tif')
##imag = cv2.split(imag)
'''
z=1
samplepath = gb.glob('C:/work/sample/*.tif')
for i in samplepath:
    imag = cv2.imread(i)
    imag = cv2.split(imag)
    w0,h0 = imag[0].shape
    c = 0
    im = Image.new('RGB',(h0,w0))
    for y in range(h0):
        for x in range(w0):
            b = int(imag[0][x,y])
            g = int(imag[1][x,y])
            r = int(imag[2][x,y])
            if b>=200 and g>=200 and r>=200:
                c = (b,g,r)
            else:
                c=(0,0,0)
            im.putpixel((y,x),c)
    im.save('oneduck200.tif')
    ##28*48
    reimage = cv2.imread('oneduck200.tif')
    bb=[]
    gg=[]
    rr=[]
    bbb=0
    ggg=0
    rrr=0
    for y in range(0,h0,2):
        for x in range(0,w0,2):
            a = reimage[x:x+2 ,y:y+2]
            aaa = np.reshape(a,(3,1,4))
            bbb = aaa[0,0].sum()
            bb.append(bbb)
            ggg = aaa[1,0].sum()
            gg.append(ggg)
            rrr = aaa[2,0].sum()
            rr.append(rrr)
    bb = np.array(bb,int)
    gg = np.array(gg,int)
    rr = np.array(rr,int)
    bb = np.reshape(bb//4,(25,25))
    gg = np.reshape(gg//4,(25,25))
    rr = np.reshape(rr//4,(25,25))
    w1,h1 = bb.shape
    im = Image.new('RGB',(w1,h1))
    for y in range(w1):
        for x in range(h1):
            r = int(rr[y,x])
            g = int(gg[y,x])
            b = int(bb[y,x])
            c = (r,g,b)
            im.putpixel((x,y),c)
    im.save('reduck.tif')
    reduck = cv2.imread('reduck.tif')
    cv2.imwrite('C:/work/resize/%d.tif'%int(z),reduck)

    reduck = cv2.split(reduck)
    w2,h2 = reduck[0].shape
    im = Image.new('L',(h2,w2))
    for y in range(h2):
        for x in range(w2):
            b = int(reduck[0][x,y])
            g = int(reduck[1][x,y])
            r = int(reduck[2][x,y])
            if b>=200 and g>=200 and r>=200:
                c = 255
            else:
                c = 0
            im.putpixel((y,x),c)
    im.save('binreduck.tif')
    binreduck = cv2.imread('binreduck.tif')
    cv2.imwrite('C:/work/binreduck/%d.tif'%int(z),binreduck)
    z=z+1

'''
'''
duck170 = cv2.imread('duck180.tif')

duck170 = cv2.split(duck170)
w2,h2 = duck170[0].shape
print(w2,h2)

im = Image.new('L',(h2,w2))
for y in range(h2):
    for x in range(w2):
        b = int(duck170[0][x,y])
        g = int(duck170[1][x,y])
        r = int(duck170[2][x,y])
        if b>=180 and g>=180 and r>=180:
            c = 255
        else:
            c = 0
        im.putpixel((y,x),c)
im.save('binduck180.tif')
'''
'''
reduck = cv2.imread('reduck.tif')
##cv2.imshow('a',reduck)
reduck = cv2.split(reduck)
w2,h2 = reduck[0].shape
print(w2,h2)

im = Image.new('L',(h2,w2))
for y in range(h2):
    for x in range(w2):
        b = int(reduck[0][x,y])
        g = int(reduck[1][x,y])
        r = int(reduck[2][x,y])
        if b>=220 and g>=220 and r>=220:
            c = 255
        else:
            c = 0
        im.putpixel((y,x),c)
im.save('binreduck.tif')

'''
############################################
'''
kernel = np.ones((5,5))

binduck180 = cv2.imread('binduck180.tif',0)
binreduck1 = cv2.imread('C:/work/binreduck1/1.tif',0)
binreduck1 = binreduck1//255
tophat1 = cv2.morphologyEx(binduck180, cv2.MORPH_TOPHAT, binreduck1)
tophat1 = binduck180 -tophat1
cv2.imwrite('tophat01.tif',tophat1)

binreduck2 = cv2.imread('C:/work/binreduck1/2.tif',0)
binreduck2 = binreduck2//255
tophat2 = cv2.morphologyEx(binduck180, cv2.MORPH_TOPHAT, binreduck2)
tophat2 = binduck180 -tophat2
cv2.imwrite('tophat02.tif',tophat2)

binreduck3 = cv2.imread('C:/work/binreduck1/3.tif',0)
binreduck3 = binreduck3//255
tophat3 = cv2.morphologyEx(binduck180, cv2.MORPH_TOPHAT, binreduck3)
tophat3 = binduck180 -tophat3
cv2.imwrite('tophat03.tif',tophat3)

binreduck4 = cv2.imread('C:/work/binreduck1/4.tif',0)
binreduck4 = binreduck4//255
tophat4 = cv2.morphologyEx(binduck180, cv2.MORPH_TOPHAT, binreduck4)
tophat4 = binduck180 -tophat4
cv2.imwrite('tophat04.tif',tophat4)

tophat5 = tophat1+tophat2+tophat3+tophat4
cv2.imwrite('tophat5_180.tif',tophat5)

tophat5 = cv2.imread('tophat5_190.tif',0)
im = Image.new('RGB',(h,w))
for y in range(h):
    for x in range(w):
        pix = int(tophat5[x,y])
        b = int(image[0][x,y])
        g  = int(image[1][x,y])
        r = int(image[2][x,y])
        if pix>0:
            c = (b,g,r)
        else:
            c = (0,0,0)
        im.putpixel((y,x),c)
im.save('duckimage190_1.tif')
'''
'''
image = cv2.imread('C:/work/full_duck.jpg')
tophat5 = cv2.imread('tophat5_180.tif',0)
image1 ,contours,hierarchy = cv2.findContours(tophat5,cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = int(cv2.contourArea(cnt))
##    print(area,len(cnt[:,0]))

    if area>=0 and area <=600:
        if int(len(cnt[:,0])) >=100 and int(len(cnt[:,0]))<200:
            ell = cv2.fitEllipse(cnt)
            image = cv2.ellipse(image,ell,(0,0,255),1)
        elif int(len(cnt[:,0])) >=5 and int(len(cnt[:,0]))<100:
            ell = cv2.fitEllipse(cnt)#
            image = cv2.ellipse(image,ell,(255,0,0),1)#
cv2.imwrite('ellipse240.tif',image) 
'''
'''
w5,h5 = tophat5.shape
im5 = Image.new("RGB",(h5,w5))
for y in range(h5):
    for x in range(w5):
        pixel = int(tophat5[x,y])
        r = int(image[2][x,y])
        g = int(image[1][x,y])
        b = int(image[0][x,y])
        if pixel >0 :
            c = (0,0,255)
        else :
            c=(r,g,b)
        im5.putpixel((y,x),c)
im5.save('paste190.tif')

'''


        

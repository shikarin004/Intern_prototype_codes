import cv2
import numpy as np 
imput_img = '/home/chanakya/Documents/Intern/Imd/12.jpg'
ori = cv2.imread(imput_img)
img = cv2.imread(imput_img)
scale_p=25
width=int(img.shape[1]*scale_p/100)
height=int(img.shape[0]*scale_p/100)
dim=(width,height)
img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
print('resized Dimensions:',img.shape)
cv2.imshow("Resized image",img)
cv2.waitKey(0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('Harris',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
corners = cv2.goodFeaturesToTrack(gray,20,0.01,10)
corners = np.int0(corners) 
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1) 
cv2.imshow('Shi-Tomasi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# sift = cv2.SIFT_create()
# kp, des = sift.detectAndCompute(gray,None)
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
# cv2.imshow('SIFT',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
# surf = cv2.xfeatures2d.SURF_create(400)
# kp, des = surf.detectAndCompute(img,None)
# img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
# cv2.imshow('Original', ori)
# cv2.imshow('SURF', img2)

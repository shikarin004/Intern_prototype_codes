import cv2
from cv2 import Canny
import numpy as np
import matplotlib.pyplot as plt
import sys
path=r'/home/chanakya/Documents/Intern/Imd/12.jpg'
img=cv2.imread(path)
window_name='Image'
kernel=np.ones((6,6),np.uint8)
image=cv2.erode(img,kernel,cv2.BORDER_REFLECT)
print('Original_d:',img.shape)
scale_p=25
width=int(img.shape[1]*scale_p/100)
height=int(img.shape[0]*scale_p/100)
dim=(width,height)
img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
print('resized Dimensions:',img.shape)
cv2.imshow("Resized image",img)
cv2.waitKey(0)

#Gaussian Blur
Gaussian=cv2.GaussianBlur(img,(15,15),0)
cv2.imshow('Gaussian Blurring',Gaussian)
cv2.waitKey(0)
#Median Blur
median=cv2.medianBlur(Gaussian,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)
#Bilateral Blur
bilateral=cv2.bilateralFilter(median,9,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('G',gray)
cv2.waitKey(0)
#Binary Inversion
_, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)
cv2.imshow('BO', bw)
cv2.waitKey(0)
edges=cv2.Canny(Gaussian,18,12,17)
cv2.imshow('canny',edges)
cv2.waitKey(0)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))

dilated = cv2.dilate(edges,kernel)
cv2.imshow('dia',dilated)
cv2.waitKey(0)

# dst=cv2.Laplacian(bw,ddepth=cMORPH_CLOSEv2.CV_16S,ksize=3)
# cv2.imshow("lap",dst)
# cv2.waitKey(0)
#Mophological Noise removal
# kernel=np.ones((3,3),np.uint8)
# closing=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,kernel,iterations=2)
# bg=cv2.dilate(closing,kernel,iterations=1)
# dit_transform=cv2.distanceTransform(closing,cv2.DIST_L2,0)
# ret,fg=cv2.threshold(dit_transform,0.02*dit_transform.max(),255,0)
# cv2.imshow('image',fg)
# cv2.waitKey(0)
# a_fg=cv2.convertScaleAbs(fg)
#Mask=cv2.bitwise_not(bw)
contours, _ = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
for i, c in enumerate(contours):
  area = cv2.contourArea(c)
  if    area< 800 or 1800 < area:    
    continue
  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  center = (int(rect[0][0]),int(rect[0][1])) 
  width = int(rect[1][0])
  height = int(rect[1][1])
  angle = int(rect[2])
  cv2.drawContours(img,[box],0,(0,0,255),1)
  cv2.circle(img, (center), 1, (255, 255, 255), -1)
cv2.imshow('Output Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Final.jpg",img)
path=r'/home/chanakya/Documents/Intern/Imd/13.jpg'
img=cv2.imread(path)
window_name='Image'
kernel=np.ones((6,6),np.uint8)
image=cv2.erode(img,kernel,cv2.BORDER_REFLECT)
print('Original_d:',img.shape)
scale_p=25
width=int(img.shape[1]*scale_p/100)
height=int(img.shape[0]*scale_p/100)
dim=(width,height)
img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
print('resized Dimensions:',img.shape)
cv2.imshow("Resized image",img)
cv2.waitKey(0)

#Gaussian Blur
Gaussian=cv2.GaussianBlur(img,(15,15),0)
cv2.imshow('Gaussian Blurring',Gaussian)
cv2.waitKey(0)
#Median Blur
median=cv2.medianBlur(Gaussian,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)
#Bilateral Blur
bilateral=cv2.bilateralFilter(median,9,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('G',gray)
cv2.waitKey(0)
#Binary Inversion
_, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)
cv2.imshow('BO', bw)
cv2.waitKey(0)
edges=cv2.Canny(Gaussian,18,12,17)
cv2.imshow('canny',edges)
cv2.waitKey(0)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))

dilated = cv2.dilate(edges,kernel)
cv2.imshow('dia',dilated)
cv2.waitKey(0)

# dst=cv2.Laplacian(bw,ddepth=cMORPH_CLOSEv2.CV_16S,ksize=3)
# cv2.imshow("lap",dst)
# cv2.waitKey(0)
#Mophological Noise removal
# kernel=np.ones((3,3),np.uint8)
# closing=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,kernel,iterations=2)
# bg=cv2.dilate(closing,kernel,iterations=1)
# dit_transform=cv2.distanceTransform(closing,cv2.DIST_L2,0)
# ret,fg=cv2.threshold(dit_transform,0.02*dit_transform.max(),255,0)
# cv2.imshow('image',fg)
# cv2.waitKey(0)
# a_fg=cv2.convertScaleAbs(fg)
#Mask=cv2.bitwise_not(bw)
contours, _ = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
for i, c in enumerate(contours):
  area = cv2.contourArea(c)
  if    area< 800 or 1800 < area:    
    continue
  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  center = (int(rect[0][0]),int(rect[0][1])) 
  width = int(rect[1][0])
  height = int(rect[1][1])
  angle = int(rect[2])
  cv2.drawContours(img,[box],0,(0,0,255),1)
  cv2.circle(img, (center), 1, (255, 255, 255), -1)
cv2.imshow('Output Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Final1.jpg",img)
from matplotlib import pyplot as plt
     
# Read the training and query images
query_img = cv2.imread('Final.jpg') 
train_img = cv2.imread('Final1.jpg') 
 
# Convert the images to grayscale 
query_img_gray = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
 
# Initialize the ORB detector algorithm 
orb = cv2.ORB_create() 
 
# Detect keypoints (features) cand calculate the descriptors
query_keypoints, query_descriptors = orb.detectAndCompute(query_img_gray,None) 
train_keypoints, train_descriptors = orb.detectAndCompute(train_img_gray,None) 
 
# Match the keypoints
matcher = cv2.BFMatcher() 
matches = matcher.match(query_descriptors,train_descriptors) 
 
# Draw the keypoint matches on the output image
output_img = cv2.drawMatches(query_img, query_keypoints, 
train_img, train_keypoints, matches[:20],None) 
 
output_img = cv2.resize(output_img, (1200,650)) 
 
# Save the final image 
cv2.imwrite("feature_matching_result.jpg", output_img) 
 
# Close OpenCV upon keypress
cv2.waitKey(0)
cv2.destroyAllWindows()


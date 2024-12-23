import cv2
import numpy as np
import random
import numpy.linalg as alg


def metric(image, x, y, ker_size=3):
    subimage = image[x - ker_size // 2: x + ker_size // 2 + 1, y - ker_size // 2: y + ker_size // 2 + 1].astype(np.float64)
    if subimage.shape != (ker_size, ker_size):
        return  0
    mean = 0
    for s in subimage:
        for p in s:
            mean += p
    mean /= ker_size**2
    print(f'{subimage = }')
    std = (subimage-mean).sum()

    print(f'{std =}')

    # std = np.std(subimage)
    subimage -= mean
    subimage /= std
    # return np.sum(subimage)

    return subimage


ker_size = 3
x,y = 200,200
image = cv2.imread('Rainier1.png')
image_gray = cv2.imread('Rainier1.png',0)
image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
image_gray = cv2.resize(image_gray, (0,0), fx=0.3, fy=0.3)

m = metric(image,x, y)
# print(m)
# print('квадрат', m**2)
# print(image.shape)
#
# cv2.imshow('image', image)
# cv2.imshow('123',image[x - ker_size // 2: x + ker_size // 2 + 1, y - ker_size // 2: y + ker_size // 2 + 1].astype(np.float64))
# cv2.waitKey(0)
# subimage = image[x - ker_size // 2: x + ker_size // 2 + 1, y - ker_size // 2: y + ker_size // 2 + 1].astype(np.float64)
# print(f'{subimage = }')
# print(f'{subimage.mean() = }')
# print(f'{subimage.std() = }')
# print(f'{(subimage - subimage.mean())/np.sqrt(subimage.std()) = }')


corners = cv2.cornerHarris(image_gray, 2, 3, 0.04)
# corners_dilate = cv2.cornerHarris(image_gray, 2, 3, 0.04)
corners_dilate = cv2.dilate(corners, None)

img1_features = image.copy()
img1_features[corners > 0.05 * corners.max()] = [0, 0, 255]

img1_features_dilate = image.copy()
img1_features_dilate[corners > 0.05 * corners.max()] = [255, 0, 255]


cv2.imshow('img1_features', img1_features)
cv2.imshow('img1_features_dilate', img1_features_dilate)

cv2.waitKey(0)


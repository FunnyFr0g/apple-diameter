import cv2
import numpy as np

image1 = cv2.imread('data/Rainier1.png')
gray_image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY).astype(np.float32)
image2 = cv2.imread('data/Rainier2.png')
gray_image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY).astype(np.float32)

# result is dilated for marking the corners, not important
img1_features = cv2.cornerHarris(gray_image1,2,3,0.04)
img1_features_dilate = cv2.dilate(img1_features, None)

img2_features = cv2.cornerHarris(gray_image2,2,3,0.04)
img2_features = cv2.dilate(img2_features, None)

feature_threshold = 0.5

def metric(image, x, y, ker_size):
    subimage = image[x-ker_size//2: x+ker_size//2+1][y-ker_size//2: y+ker_size//2+1]
    return np.sum(subimage)

def feature_to_list(img,matrix):
    result = []
    for i in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            d = metric(img, i, k, 3)
            result.append((d,i,k))

    return result


img1_features_list = feature_to_list(gray_image1, img1_features)
img2_features_list = feature_to_list(gray_image2, img2_features)

good_features_list = [] # (metric, x,y)

for i, f1 in enumerate(img1_features_list):

    for f2 in img2_features_list:
        if np.abs(f1[0]-f2[0]) < feature_threshold:
            print(f'Отбрасывание фич {round(i / len(img1_features_list) * 100, 2)}%')
            good_features_list.append(f1)
            good_features_list.append(f2)

print(good_features_list)

def solve_affine( p1, p2, p3, p4, s1, s2, s3, s4 ):
    x = np.transpose(np.matrix([p1,p2,p3,p4]))
    y = np.transpose(np.matrix([s1,s2,s3,s4]))
    # add ones on the bottom of x and y
    x = np.vstack((x,[1,1,1,1]))
    y = np.vstack((y,[1,1,1,1]))
    # solve for A2
    A2 = y * x.I
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is
    return lambda x: (A2*np.vstack((np.matrix(x).reshape(3,1),1)))[0:3, :]

print(solve_affine(1,2,3,4,1,2,3,4))


# def ransac(img1, img2, features):










print(f"{image1.shape = }")
print(f"{len(img1_features) = }")

#Нарисуем эти фичиб наложив маски
image1[img1_features > 0.01 * img1_features.max()] = [0, 0, 255]
image2[img2_features > 0.01 * img2_features.max()] = [0, 0, 255]

cv2.imshow('image1', img1_features)
cv2.imshow('image2', img1_features_dilate)

cv2.waitKey(0)
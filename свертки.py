from pickletools import uint8

import cv2
import numpy as np

image = cv2.imread('lab4/data/dog.jpg')# , cv2.IMREAD_GRAYSCALE)
w, h = image.shape[0:2]
new_image = np.zeros((w,h,3), dtype=np.uint8)

# ker = np.array([[1,1,1],  #Ближайший сосед
#                 [1,1,1],
#                 [1,1,1]])

sobel1_ker = np.array([[-1,-2,-1], # Собель
                [0,0,0],
                [1,2,1]])

sobel2_ker = np.array([[-1,0,1], # Собель
                        [-2,0,2],
                        [-1,0,1]])

gaus_ker = np.array([[0,1,0],  # гаусс 3х3
                [1,4,1],
                [0,1,0]])

ker = np.array([[-2,-1,0],  # мультяшный
                [-1,1,1],
                [0,1,2]])

sharpness_ker = np.array([[0,-1,0],  # резкость
                [-1,5,-1],
                [0,-1,0]])

def gaussian_kernel(size=7, sigma=1):
    """Создание ядра свертки Гаусса размером size x size"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    ) * 1000/3
    return kernel

# ker = np.round(gaussian_kernel(size=7, sigma=1))

print(ker)
ker = gaus_ker
ker_w, ker_h = ker.shape


her = image[30:33, 30:33]
print(her, '-'*20)
# print((ker*her).sum(axis=(0, 1))/9)

# print(type(ker))

def my_local_conv(a,b):
    if len(b.shape) == 2:
        result = 0
    else:
        result = np.array((0,0,0))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # print(result,'+')
            result += a[i,j]*b[i,j]

    return result


def my_conv(K, M):
    normalizer = K.sum()
    if normalizer == 0:
        normalizer = 1
    b = my_local_conv(K, M)
    result = b/normalizer

    def to_8bit(a):
        if a>255:
            return 255
        if a<0:
            return 0
        return int(a)

    # return result
    return [to_8bit(elem) for elem in result]

# print(my_conv(ker, her))

for i in range(ker_w//2, w -ker_w//2-1):
    print(round(i/w*100),'%')
    for j in range(ker_h//2, h-ker_h//2-1):
        im_crop = (image[i-ker_w//2:i+ker_w//2+1, j-ker_h//2:j+ker_h//2+1]).astype(np.uint16)
        new_image[i,j] = my_conv(ker, im_crop)

print(image[30:40, 30:40])
print(new_image[30:40, 30:40])

def image_decrease(img, n):
    if not n%2:
        n += 1
    h, w = img.shape[:2]
    new_img = np.zeros((h//n,w//n,3), dtype=np.uint8)
    for col in range(new_img.shape[1]):
        print('Уменьшение картинки: ',round(col/w*100),'%')
        for row in range(new_img.shape[0]):
            new_img[row][col] = img[row*n+n//2][col*n+n//2]
    return new_img

small_img = image_decrease(image,5)
small_img_gauss = image_decrease(new_image,5)

# new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('original', image)
cv2.imshow('my_ker', new_image)
cv2.imshow('small_image', small_img)
cv2.imshow('small_image_gauss', small_img_gauss)


cv2.waitKey(0)

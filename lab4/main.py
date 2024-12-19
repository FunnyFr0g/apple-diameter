import cv2
import numpy as np

image = cv2.imread('data/dogsmall.jpg', cv2.IMREAD_GRAYSCALE)

# image = cv2.resize(image, (50, 50))
# image = image.resize()
N, M = image.shape[0:2]

print(type(image))

def furier_image(k, l):
    result = 0
    for i in range(N):
        for j in range(M):
            result += image[i,j] * np.exp(-2j*np.pi*(k*i/N+l*j/M))
    return result


dft_image = cv2.dft(image.astype(np.float32))
# dft_image += 1
# cv2.log(dft_image,dft_image)
#
# cx = int(N/2)
# cy = int(M/2)
# q0 = dft_image[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
# q1 = dft_image[cx:cx+cx, 0:cy]     # Top-Right
# q2 = dft_image[0:cx, cy:cy+cy]     # Bottom-Left
# q3 = dft_image[cx:cx+cx, cy:cy+cy]
#
# tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
# dft_image[0:cx, 0:cy] = q3
# dft_image[cx:cx + cx, cy:cy + cy] = tmp
# tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
# dft_image[cx:cx + cx, 0:cy] = q2
# dft_image[0:cx, cy:cy + cy] = tmp
#
#
# cv2.normalize(dft_image,dft_image,0,1,cv2.NORM_MINMAX)
#
# cv2.imshow('DFT',dft_image)
# cv2.waitKey(0)


print(dft_image)

my_dft_image = np.zeros(image.shape)

for k in range(N):
    print(f'{round(k / N * 100)}%')
    for l in range(M):
        my_dft_image[k,l] = np.abs(furier_image(k, l))


my_dft_image = cv2.log(1 + my_dft_image)
# cv2.normalize(my_dft_image, my_dft_image, 0, 1, cv2.NORM_MINMAX)
my_dft_image /= my_dft_image.max()
my_dft_image *= 255
my_dft_image = my_dft_image.astype(np.uint8)

# m = np.max(dft_image)
# dft_image = dft_image/m*255

print(my_dft_image)
cv2.imwrite('mydft_image.jpg', my_dft_image)
cv2.imshow('1234',image)
cv2.imshow('my_dft', my_dft_image)
cv2.imshow('dft', dft_image)
cv2.imwrite('dft_image.jpg', dft_image)

cv2.waitKey(0)
import cv2
import numpy as np


img = cv2.imread("Rainier1.png",0)
patch = img[150:166, 200:216].astype(np.float64)

gx= cv2.Sobel(patch, ddepth=-1, dx=1, dy=0, ksize = 3) #градиенты
print(f'{gx.shape = }')
gy = cv2.Sobel(patch, ddepth=-1, dx=0, dy=1, ksize = 3) #градиенты


# if np.all(gx == 0) and np.all(gy == 0):

magnitude = np.sqrt(gx * gx + gy * gy) #величина градиента
orientation = np.arctan2(gy, gx) * (180 / np.pi) % 360 #направление градиента

descriptor = np.histogram(orientation, bins=8, range=(0, 360), weights=magnitude)[0]

norm = np.linalg.norm(descriptor)
if norm == 0:
    descriptor = np.zeros_like(descriptor)  # Если норма нулевая, дескриптор остается нулевым
else:
    descriptor = descriptor / norm  # Нормализация
print(magnitude)
print(np.array(descriptor))

cv2.imshow('img', img[200:216, 200:216])
cv2.waitKey(0)
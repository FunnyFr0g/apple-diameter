import cv2
import numpy as np

image = cv2.imread("cat.png")

w = image.shape[0]
h = image.shape[1]

image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


for row in range(w):
    for column in range(h):

        bgr_pixel = np.array(image[row,column], dtype=np.float32)
        B, G, R = bgr_pixel

        V = max(R,G,B)
        if V>0:
            S = (V - min(bgr_pixel))/V
        else:
            S = 0

        H = 0
        if R == G == B:
            H = 0
        elif V == R:
            H = 60*(G-B)/(V-min(R,G,B))
        elif V == G:
            H = 120 + 60*(B-R)/(V-min(R,G,B))
        elif V == B:
            H = 120 + 60*(R-G)/(V-min(R,G,B))

        H%=360


        image[row, column] = (H/2, S*255, V) # чтобы уложиться в 8-битный формат




image2[1] += 100

print('Мой результат',image[1:10,100])
print("Результат cv2",image2[1:10,100])
print(image.shape)


cv2.imshow('my', image)
cv2.imshow('cv2', image2)

#cv2.imwrite("test.png",image)
cv2.waitKey(0)

#print(image[100,100]) #NOT RGB - BGR


''' и мы поигрались с кодом
TASK 1 

'''
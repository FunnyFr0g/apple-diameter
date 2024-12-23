import cv2
import numpy as np
import random
import numpy.linalg as alg
import os

from torchvision.transforms.v2.functional import affine


def metric(image, x, y, ker_size):
    subimage = image[x - ker_size // 2: x + ker_size // 2 + 1, y - ker_size // 2: y + ker_size // 2 + 1].astype(np.float64)
    if subimage.shape != (ker_size, ker_size):
        return  None

    # return subimage
    return (subimage - subimage.mean())/np.sqrt(subimage.std())

def feature_to_list(img, matrix, harris_threshold=0.05):
    result = []
    print(f'{matrix[1,1] = }')
    m = np.max(matrix)
    print(f'{m = }')
    for i in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[i, k] > harris_threshold*m:
                d = metric(img, i, k, 5)
                if d is not None :
                    result.append((d, i, k))

    features = img[matrix > harris_threshold*m] = 255
    cv2.imshow('features', matrix)
    cv2.waitKey(0)

    return result


def get_matched_features(im1, im2, feature_match_threshold=100):
    """Сравниваем попарно дескрипторы. Находим минимальную разность, меньшую порога treshold.
    Выводим список фич, где на четном месте фичи с картинки1, на нечетном - с картинки 2.
    Каждая запись содержит (дескриптор, X, Y) """

    img1_features = cv2.cornerHarris(gray_image1, 2, 3, 0.04)
    # img1_features = cv2.dilate(img1_features, None)  # Подавление немаксимумов

    img2_features = cv2.cornerHarris(gray_image2, 2, 3, 0.04)
    # img2_features = cv2.dilate(img2_features, None)

    img1_features_list = feature_to_list(gray_image1, img1_features)
    img2_features_list = feature_to_list(gray_image2, img2_features)

    good_features_list = []  # (metric, x,y)

    for i, f1 in enumerate(img1_features_list):
        best = float("inf")
        print(f'Мэтчинг фич {round(i / len(img1_features_list) * 100, 2)}%, {len(good_features_list) = }')
        for f2 in img2_features_list:
            # dist = np.abs(f1[0] - f2[0])
            # dist = np.sum(np.abs(f1[0] - f2[0]))
            dist = np.sum((f1[0] - f2[0])**2)
            if dist < feature_match_threshold and dist < best:
                best_feature = f2
                best = dist

        if best < float("inf"):
            good_features_list.append(f1)
            good_features_list.append(best_feature)

    return good_features_list

def ransac(img1, img2, good_features: np.ndarray, iterations=1000, inliners_edge=5):

    f1_list = good_features[::2]
    f2_list = good_features[1::2]

    if len(f1_list) <= 3:
        print(f'Нашлось всего {len(f1_list)} фич, этого недостаточно')
        return None
    index_list = [i for i in range(len(f1_list))]
    best_inliners = 0

    for _ in range(iterations):
        print(f'RANSAC {round(_/iterations*100, 2)}%, {best_inliners = }')
        inliners = 0
        random.shuffle(index_list)
        m1, m2, m3 = index_list[0:3]  # возьмем 3 случайных уникальных индекса
        mx1, my1, mx2, my2, mx3, my3 = f1_list[m1][2], f1_list[m1][1], f1_list[m2][2], f1_list[m2][1], f1_list[m3][2], f1_list[m3][1]  # получим координаты этих точек

        M = np.array([[mx1, my1, 1, 0, 0, 0],
                      [0, 0, 0, mx1, my1, 1],
                      [mx2, my2, 1, 0, 0, 0],
                      [0, 0, 0, mx2, my2, 1],
                      [mx3, my3, 1, 0, 0, 0],
                      [0, 0, 0, mx3, my3, 1]], dtype=np.float64)

        n1, n2, n3 = m1, m2, m3  # Это просто индексы. Поскольку соответствующие фичи из im1 и im2 идут в одинаковом порядке, они совпадают.
        nx1, ny1, nx2, ny2, nx3, ny3 = f2_list[n1][2], f2_list[n1][1], f2_list[n2][2], f2_list[n2][1], f2_list[n3][2], f2_list[n3][1]
        b = np.array([nx1,
                          ny1,
                          nx2,
                          ny2,
                          nx3,
                          ny3], dtype=np.float64)

        if alg.det(M) != 0:
            a = np.matmul(alg.inv(M), b)
        else:
            continue

        for f1, f2 in zip(f1_list, f2_list):
            n_x = a[0] * f1[2] + a[1] * f1[1] + a[2]
            n_y = a[3] * f1[2] + a[4] * f1[1] + a[5]
            if np.sqrt((n_x - f2[2]) ** 2 + (n_y - f2[1]) ** 2) < inliners_edge:
                inliners += 1

        if inliners > best_inliners:
            best_inliners = inliners
            best_matrix = a
    print(f'{best_matrix = }')
    return best_matrix

def stitch_images(image1, image2, best_matrix):
    new_positions = []
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), -float("inf"), -float("inf"),
    for i in range(image1.shape[0]):
        row = []
        for j in range(image1.shape[1]):
            n_x = round(best_matrix[0] * j + best_matrix[1] * i + best_matrix[2])
            n_y = round(best_matrix[3] * j + best_matrix[4] * i + best_matrix[5])
            if n_x < min_x:
                min_x = n_x
            if n_y < min_y:
                min_y = (n_y)
            if n_x > max_x:
                max_x = (n_x)
            if n_y > max_y:
                max_y = n_y
            row.append((n_y, n_x))
        new_positions.append(row)
    
    shift_x = 0
    shift_y = 0
    if int(min_x) < 0:
        shift_x = -int(min_x)
    if int(min_y) < 0:
        shift_y = -int(min_y)

    print(f'{shift_y = }, {shift_x = }')
    # print(new_positions[0][0])
    # print(new_positions[image1.shape[0] - 1][image1.shape[1] - 1])
    # print(max_x - min_x + image2.shape[1])

    mixed_image = np.zeros(shape=(image1.shape[0] + image2.shape[0], image1.shape[1] + image2.shape[1] + 1, 3),
                           dtype=np.uint8)
    # отрисуем первую картинку
    for y in range(image2.shape[0]):
        for x in range(image2.shape[1]):
            mixed_image[y][x] = image2[y][x]

    # отрисуем вторую картинку после преобразования
    for y in range(image1.shape[0]):
        for x in range(image1.shape[1]):
            mixed_image[new_positions[y][x][0] + shift_y][new_positions[y][x][1] + shift_x] = image1[y][x]

    return mixed_image


####################################################
    # print(f"{best_matrix = }, {best_inliners = }")
    # x_news = []
    # y_news = []
    #
    # for y in range(image1.shape[0]):  # новые координаты для пикселей
    #     for x in range(image1.shape[1]):
    #         n_x = round(best_matrix[0] * x + best_matrix[1] * y + best_matrix[2])
    #         n_y = round(best_matrix[3] * x + best_matrix[4] * y + best_matrix[5])
    #         x_news.append(n_x)
    #         y_news.append(n_y)
    #
    # x_min, x_max = min(x_news), max(x_news)
    # y_min, y_max = min(y_news), max(y_news)
    #
    # shift_x = -x_min if x_min < 0 else 0
    # shift_y = -y_min if y_min < 0 else 0
    #
    # # mixed_image = np.zeros(shape=(image1.shape[0] + image2.shape[0], image1.shape[1] + image2.shape[1] + 1, 3), dtype=np.uint8)
    # mixed_image = np.zeros(shape=(image1.shape[0]*5, image1.shape[1]*5 , 3),
    #                        dtype=np.uint8)
    # # mixed_image[0:image2.shape[0], 0:image2.shape[1]] = image2
    # # for y in range(image2.shape[0]):
    # #     for x in range(image2.shape[1]):
    # #         mixed_image[y][x] = image2[y][x]
    #
    # for y in range(image1.shape[0]):
    #     for x in range(image1.shape[1]):
    #         mixed_image[y_news[y] + shift_y][x_news[x] + shift_x] = image1[y][x]
    #
    # return mixed_image



image1 = cv2.imread('data/Rainier2.png')
# image1 = cv2.resize(image1, (0, 0), fx = 0.3, fy = 0.3)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)

image2 = cv2.imread('data/Rainier1.png')
# image2 = cv2.resize(image2, (0, 0), fx = 0.3, fy = 0.3)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)

features_list = get_matched_features(gray_image1, gray_image2)
affine_matrix = ransac(gray_image1, gray_image2, features_list)
mixed_image = stitch_images(image1, image2, affine_matrix)

cv2.imshow("mix", mixed_image)
cv2.imwrite("mix.png", mixed_image)
cv2.waitKey(0)


# print(f"{image1.shape = }")
# print(f"{len(img1_features) = }")
#
# #Нарисуем эти фичиб наложив маски
# image1[img1_features > 0.01 * img1_features.max()] = [0, 0, 255]
# image2[img2_features > 0.01 * img2_features.max()] = [0, 0, 255]
#
# cv2.imshow('image1', img1_features)
# cv2.imshow('image2', img1_features_dilate)
#
# cv2.waitKey(0)

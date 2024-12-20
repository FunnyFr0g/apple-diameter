import cv2
import numpy as np
import random
import numpy.linalg as alg


def metric(image, x, y, ker_size):
    subimage = image[x - ker_size // 2: x + ker_size // 2 + 1, y - ker_size // 2: y + ker_size // 2 + 1]
    if subimage.shape != (ker_size, ker_size):
        return  0
    print(subimage.shape)
    mean = 0
    for s in subimage:
        for p in s:
            mean += p
    mean /= ker_size**2
    std = sum((subimage-mean)**2)

    # std = np.std(subimage)
    subimage -= mean
    subimage /= std
    # return np.sum(subimage)

    return subimage


def feature_to_list(img, matrix):
    result = []
    m = np.max(matrix)
    for i in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[i, k] > 0.01*m:
                d = metric(img, i, k, 3)
                if d is not 0 :
                    result.append((d, i, k))

    return result


def get_compared_features(im1, im2, feature_threshold=0.05):
    """Сравниваем попарно дескрипторы. Находим минимальную разность, меньшую порога treshold.
    Выводим список фич, где на четном месте фичи с картинки1, на нечетном - с картинки 2.
    Каждая запись содержит (дескриптор, X, Y) """

    img1_features = cv2.cornerHarris(gray_image1, 2, 3, 0.04)
    img1_features = cv2.dilate(img1_features, None)  # Подавление немаксимумов

    img2_features = cv2.cornerHarris(gray_image2, 2, 3, 0.04)
    img2_features = cv2.dilate(img2_features, None)

    img1_features_list = feature_to_list(im1, img1_features)
    img2_features_list = feature_to_list(im2, img2_features)

    good_features_list = []  # (metric, x,y)

    for i, f1 in enumerate(img1_features_list):
        best = float("inf")
        print(f'Отбрасывание фич {round(i / len(img1_features_list) * 100, 2)}%, {len(good_features_list) = }')
        for f2 in img2_features_list:
            # dist = np.abs(f1[0] - f2[0])
            dist = np.sum(np.abs(f1[0] - f2[0]))
            if dist < feature_threshold and dist < best:
                best_feature = f2
                best = dist

        if best < float("inf"):
            good_features_list.append(f1)
            good_features_list.append(best_feature)

    return good_features_list


def solve_affine(p1, p2, p3, p4, s1, s2, s3, s4):
    x = np.transpose(np.matrix([p1, p2, p3, p4]))
    y = np.transpose(np.matrix([s1, s2, s3, s4]))
    # add ones on the bottom of x and y
    x = np.vstack((x, [1, 1, 1, 1]))
    y = np.vstack((y, [1, 1, 1, 1]))
    # solve for A2
    A2 = y * x.I
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is
    return lambda x: (A2 * np.vstack((np.matrix(x).reshape(3, 1), 1)))[0:3, :]


def ransac(img1, img2, good_features: np.ndarray, iterations=1000, inliners_edge=10):

    f1_list = good_features[::2]
    f2_list = good_features[1::2]

    if len(f1_list) <= 3:
        print(f'Нашлось всего {len(f1_list)} фич, этого недостаточно')
        return None
    index_list = [i for i in range(len(f1_list))]
    best_inliners = 0

    for _ in range(iterations):
        inliners = 0
        random.shuffle(index_list)
        m1, m2, m3 = index_list[0:3]  # возьмем 3 случайных уникальных индекса
        mx1, my1, mx2, my2, mx3, my3 = f1_list[m1][1], f1_list[m1][2], f1_list[m2][1], f1_list[m2][2], f1_list[m3][1], f1_list[m3][2]  # получим координаты этих точек

        M = np.array([[mx1, my1, 1, 0, 0, 0],
                      [0, 0, 0, mx1, my1, 1],
                      [mx2, my2, 1, 0, 0, 0],
                      [0, 0, 0, mx2, my2, 1],
                      [mx3, my3, 1, 0, 0, 0],
                      [0, 0, 0, mx3, my3, 1]], dtype=np.float64)

        n1, n2, n3 = m1, m2, m3  # Это просто индексы. Поскольку соответствующие фичи из im1 и im2 идут в одинаковом порядке, они совпадают.
        nx1, ny1, nx2, ny2, nx3, ny3 = f2_list[n1][1], f2_list[n1][2], f2_list[n2][1], f2_list[n2][2], f2_list[n3][1], f2_list[n3][2]
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
            n_x = a[0] * f1[1] + a[1] * f1[2] + a[2]
            n_y = a[3] * f1[1] + a[4] * f1[2] + a[5]
            if np.sqrt((n_x - f2[1]) ** 2 + (n_y - f2[2]) ** 2) < inliners_edge:
                inliners += 1

        if inliners > best_inliners:
            best_inliners = inliners
            best_matrix = a

    print(f"{best_matrix = }, {best_inliners = }")
    x_news = []
    y_news = []

    for y in range(image1.shape[0]):  # новые координаты для пикселей
        for x in range(image2.shape[1]):
            n_x = int(best_matrix[0] * x + best_matrix[1] * y + best_matrix[2])
            n_y = int(best_matrix[3] * x + best_matrix[4] * y + best_matrix[5])
            x_news.append(n_x)
            y_news.append(n_y)

    x_min, x_max = min(x_news), max(x_news)
    y_min, y_max = min(y_news), max(y_news)

    shift_x = -x_min if x_min < 0 else 0
    shift_y = -y_min if y_min < 0 else 0

    mixed_image = np.zeros(shape=(image1.shape[0] + image2.shape[0], image1.shape[1] + image2.shape[1] + 1, 3),
                           dtype=np.uint8)
    # mixed_image[0:image2.shape[0], 0:image2.shape[1]] = image2
    for y in range(image2.shape[0]):
        for x in range(image2.shape[1]):
            mixed_image[y][x] = image2[y][x]

    for y in range(image1.shape[0]):
        for x in range(image1.shape[1]):
            mixed_image[y_news[y] + shift_y][x_news[x] + shift_x] = image1[y][x]

    return mixed_image



image1 = cv2.imread('data/Rainier1.png')
image1 = cv2.resize(image1, (0, 0), fx = 0.3, fy = 0.3)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)

image2 = cv2.imread('data/Rainier2.png')
image2 = cv2.resize(image2, (0, 0), fx = 0.3, fy = 0.3)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)

features_list = get_compared_features(gray_image1, gray_image2)

mixed_image = ransac(image1, image2, features_list)
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

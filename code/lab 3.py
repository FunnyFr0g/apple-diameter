import cv2
import numpy as np
import random
import numpy.linalg as alg


def metric(image, x, y, ker_size):
    subimage = image[x - ker_size // 2: x + ker_size // 2 + 1, y - ker_size // 2: y + ker_size // 2 + 1].astype(np.float64)
    if subimage.shape != (ker_size, ker_size):
        return  None

    return subimage


def feature_to_list(img, matrix):
    result = []
    m = np.max(matrix)
    for i in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[i, k] > 0.01*m:
                d = metric(img, i, k, 3)
                if d is not None :
                    result.append((d, i, k))

    return result


def get_matched_features(im1, im2, feature_threshold=50):
    """Сравниваем попарно дескрипторы. Находим минимальную разность, меньшую порога treshold.
    Выводим список фич, где на четном месте фичи с картинки1, на нечетном - с картинки 2.
    Каждая запись содержит (дескриптор, X, Y) """

    img1_features = cv2.cornerHarris(gray_image1, 2, 3, 0.04)
    img1_features = cv2.dilate(img1_features, None)  # Подавление немаксимумов

    img2_features = cv2.cornerHarris(gray_image2, 2, 3, 0.04)
    img2_features = cv2.dilate(img2_features, None)

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


def ransac(img1, img2, good_features: np.ndarray, iterations=10, coridor=50):

    features_pos_1 = good_features[::2]
    features_pos_2 = good_features[1::2]

    print(f'{features_pos_1 = }')

    if len(features_pos_1) <= 3:
        print(f'Нашлось всего {len(features_pos_1)} фич, этого недостаточно')
        return None
    index_list = [i for i in range(len(features_pos_1))]
    best_inliners = 0


    find = good_features

    i = 0
    best_matrix = None
    max_popalo = 0
    index_list = [i for i in range(len(features_pos_1))]
    while i < iterations:
        print(f'RANSAC {round(i/iterations*100, 2)}%')
        random.shuffle(index_list)
        p1,p2,p3= index_list[0:3] 

        M = np.array([[features_pos_1[p1][2], features_pos_1[p1][1], 1, 0, 0, 0],
                      [0, 0, 0, features_pos_1[p1][2], features_pos_1[p2][1], 1],
                      [features_pos_1[p2][2], features_pos_1[p2][1], 1, 0, 0, 0],
                      [0, 0, 0, features_pos_1[p2][2], features_pos_1[p2][1], 1],
                      [features_pos_1[p3][2], features_pos_1[p3][1], 1, 0, 0, 0],
                      [0, 0, 0, features_pos_1[p3][2], features_pos_1[p3][1], 1]
                      ], dtype=np.float64)

        b = np.array([features_pos_2[p1][2],
                      features_pos_2[p1][1],
                      features_pos_2[p2][2],
                      features_pos_2[p2][1],
                      features_pos_2[p3][2],
                      features_pos_2[p3][1]], dtype=np.float64)
        if alg.det(M) != 0:
            i += 1
            # A = np.matmul(np.matmul(lg.inv(np.matmul(lg.matrix_transpose(M), M)), lg.matrix_transpose(M)), b)
            A = np.matmul(alg.inv(M), b)
        else:
            continue
        # Матрицей я переношу с 1-ой фото на вторую
        popalo = 0
        for j in range(len(features_pos_1)):
            n_x = A[0] * features_pos_1[j][2] + A[1] * features_pos_1[j][1] + A[2]
            n_y = A[3] * features_pos_1[j][2] + A[4] * features_pos_1[j][1] + A[5]
            if abs(n_x - features_pos_2[j][2]) < coridor and abs(n_y - features_pos_2[j][1]) < coridor:
                popalo += 1

        if popalo > max_popalo:
            max_popalo = popalo
            best_matrix = A

    new_positions = []
    min_x = float("inf")
    min_y = float("inf")
    max_x = -float("inf")
    max_y = -float("inf")
    for i in range(image1.shape[0]):
        row = []
        for j in range(image1.shape[1]):
            n_x = best_matrix[0] * j + best_matrix[1] * i + best_matrix[2]
            n_y = best_matrix[3] * j + best_matrix[4] * i + best_matrix[5]
            if int(n_x) < min_x:
                min_x = int(n_x)
            if int(n_y) < min_y:
                min_y = int(n_y)
            if int(n_x) > max_x:
                max_x = int(n_x)
            if int(n_y) > max_y:
                max_y = int(n_y)
            row.append((int(n_y), int(n_x)))
        new_positions.append(row)

    shift_x = 0
    shift_y = 0
    if int(min_x) < 0:
        shift_x = -int(min_x)
    if int(min_y) < 0:
        shift_y = -int(min_y)

    print(shift_y, shift_x)
    print(new_positions[0][0])
    print(new_positions[image1.shape[0] - 1][image1.shape[1] - 1])
    print(max_x - min_x + image2.shape[1])

    mixed_image = np.zeros(shape=(image1.shape[0] + image2.shape[0]*5, image1.shape[1] + image2.shape[1]*5 + 1),
                           dtype=np.uint8)

    for y in range(image2.shape[0]):
        for x in range(image2.shape[1]):
            mixed_image[y][x] = image2[y][x]

    for y in range(image1.shape[0]):
        for x in range(image1.shape[1]):
            mixed_image[new_positions[y][x][0] + shift_y][new_positions[y][x][1] + shift_x] = image1[y][x]

    cv2.imshow("mix", mixed_image)
    cv2.waitKey(0)



image1 = cv2.imread('Rainier2.png',0)
image1 = cv2.resize(image1, (0, 0), fx = 0.25, fy = 0.25)
# gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_image1 = image1

image2 = cv2.imread('Rainier1.png',0)
image2 = cv2.resize(image2, (0, 0), fx = 0.25, fy = 0.25)
# gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_image2 = image2

features_list = get_matched_features(gray_image1, gray_image2)

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

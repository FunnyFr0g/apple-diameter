

        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H

    return best_H



def stitch_images(image1, image2, H):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    corners = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    all_corners = np.concatenate((corners, transformed_corners), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5) #размеры итога

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    transformed_image = cv2.warpPerspective(image2, np.dot(translation_matrix, H), (x_max - x_min, y_max - y_min)) #сдвиг

    panorama = np.zeros_like(transformed_image)
    panorama[-y_min:height1 - y_min, -x_min:width1 - x_min] = image1

    panorama = np.maximum(panorama, transformed_image)

    return panorama


image_paths = ["data/Rainier1.png", "data/Rainier2.png"]#, "data/Rainier3.png", "data/Rainier4.png", "data/Rainier5.png", "data/Rainier6.png"]
imgs = []
for i in range(len(image_paths)):
    imgs.append(cv2.imread(image_paths[i]))


# keypoints0 = np.loadtxt('keypoints0_tresh_harris_0.01.txt', delimiter=',').astype(np.int64)
# keypoints1 = np.loadtxt('keypoints1_tresh_harris_0.01.txt', delimiter=',').astype(np.int64)
# # descriptors0 = np.loadtxt('descriptors0_tresh_harris_0.01.txt', delimiter=',').astype(np.float32)
# # descriptors1 = np.loadtxt('descriptors1_tresh_harris_0.01.txt', delimiter=',').astype(np.float32)
# points0 = np.loadtxt('points0_tresh_0.01.txt', delimiter=',')
# points1 = np.loadtxt('points1_tresh_0.01.txt', delimiter=',')
H = np.loadtxt('H_tresh_harris_0.01.txt', delimiter=',').astype(np.float64)

# keypoints0 = harris_corner_detector(imgs[0])
# keypoints1 = harris_corner_detector(imgs[1])
# descriptors0 = sift_descriptor(imgs[0], keypoints0)
# descriptors1 = sift_descriptor(imgs[1], keypoints1)
# matches = match_keypoints(descriptors0, descriptors1)
# points0 = np.float32([keypoints0[m[0]] for m in matches])
# points1 = np.float32([keypoints1[m[1]] for m in matches])
# H = ransac(points0, points1)

panorama = stitch_images(imgs[1], imgs[0], H)

# np.savetxt("keypoints0_tresh_harris_0.01.txt", keypoints0, fmt="%d", delimiter=",")
# np.savetxt("keypoints1_tresh_harris_0.01.txt", keypoints1, fmt="%d", delimiter=",")
# np.savetxt("descriptors0_tresh_harris_0.01.txt", descriptors0, fmt="%.18e", delimiter=",")
# np.savetxt("descriptors1_tresh_harris_0.01.txt", descriptors1, fmt="%.18e", delimiter=",")
# np.savetxt("points0_tresh_0.01.txt", points0, fmt="%d", delimiter=",")
# np.savetxt("points1_tresh_0.01.txt", points1, fmt="%d", delimiter=",")
# np.savetxt("H_tresh_harris_0.01.txt", H, fmt="%.18e", delimiter=",")



stitchy = cv2.Stitcher.create()
(dummy, output) = stitchy.stitch(imgs)

if dummy != cv2.STITCHER_OK:
    print("stitching ain't successful")
else:
    print('Your Panorama is ready!!!')



# output_image1 = imgs[0].copy()
# output_image2 = imgs[1].copy()
#
# for corner in keypoints0:
#     x, y = corner
#     if (x, y) in points0:
#         cv2.circle(output_image2, (x, y), 3, (0, 255, 0), -1)
#     cv2.circle(output_image1, (x, y), 3, (0, 0, 255), -1)
#
# for corner in keypoints1:
#     x, y = corner
#     if (x, y) in points1:
#         cv2.circle(output_image1, (x, y), 3, (0, 255, 0), -1)
#     cv2.circle(output_image2, (x, y), 3, (0, 0, 255), -1)
#
# cv2.imshow("1", output_image1)
# cv2.imshow("2", output_image2)
# cv2.waitKey(0)

cv2.imshow("result", panorama)
cv2.imshow('result cv2', output)
cv2.waitKey(0)

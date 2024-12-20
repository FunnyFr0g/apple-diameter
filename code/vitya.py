import numpy as np

def sift_descriptor(image, keypoints, patch_size=16):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    descriptors = []

    for kp in keypoints:
        x, y = kp
        height, width = image.shape

        if x - patch_size // 2 < 0 or x + patch_size // 2 >= width or y - patch_size // 2 < 0 or y + patch_size // 2 >= height:
            continue
        patch = image[y - patch_size // 2:y + patch_size // 2, x - patch_size // 2:x + patch_size // 2]


        gx, gy = sobel(patch) #градиенты

        if np.all(gx == 0) and np.all(gy == 0):
            continue

        magnitude = np.sqrt(gx * gx + gy * gy) #величина градиента
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 360 #направление градиента

        descriptor = np.histogram(orientation, bins=8, range=(0, 360), weights=magnitude)[0]

        norm = np.linalg.norm(descriptor)
        if norm == 0:
            descriptor = np.zeros_like(descriptor)  # Если норма нулевая, дескриптор остается нулевым
        else:
            descriptor = descriptor / norm  # Нормализация
        descriptors.append(descriptor)
        print(np.array(descriptors))
    return np.array(descriptors)

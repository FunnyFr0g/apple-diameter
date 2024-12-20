import cv2
import numpy as np


def harris_corner_detector(gray, window_size=3, k=0.04, threshold=0.05):
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    kernel = np.ones((window_size, window_size))
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)

    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    corners = np.zeros_like(gray)
    corners[R > threshold * R.max()] = 255
    return corners


def lucas_kanade(prev_gray, next_gray, corners, window_size=5):
    Ix = cv2.Sobel(next_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(next_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = next_gray - prev_gray

    half_window = window_size // 2
    flow = []

    corners_y, corners_x = np.where(corners > 0)

    close_vectors = []

    for y, x in zip(corners_y, corners_x):
        Ix_window = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
        Iy_window = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
        It_window = It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()

        A = np.vstack((Ix_window, Iy_window)).T
        b = -It_window

        nu = np.linalg.pinv(A) @ b
        flow.append((x, y, nu[0], nu[1]))
        # if len(flow)>0:
        #     if (abs(y-flow[-1][0]) + abs(x-flow[-1][1])) < 50:
        #         close_vectors.append((x, y, nu[0], nu[1]))
        #     else:
        #         magnitudes = []
        #         angles = []
        #         for x1, y1, x2, y2 in close_vectors:
        #             magnitudes.append(np.sqrt((y-flow[-1][0])**2 + (x-flow[-1][1])**2))
        #             angles.append(np.arctan2(x2-x1, y2-y1) * (180 / np.pi) % 360)
        #
        #         hist = np.histogram(angles, bins=8, range=(0, 360), weights=magnitudes)[0]
        #         magnitude = hist.max()
        #         angle = hist.argmax() * np.pi/4
        #
        #         u = np.cos(angle)*magnitude
        #         v = np.sin(angle)*magnitude
        #         flow.append((x, y, u, v))
        # else:
        #     flow.append((x, y, nu[0], nu[1]))


    return flow


def draw_optical_flow(frame, flow):
    for x, y, u, v in flow:
        print(f'{u = }, {v = }')
        # x1, y1 = int(x - u*100 if u>0.2 else x), int(y - v*100 if v>0.2 else y)
        x1, y1 = int(x - u*100), int(y - v*100)
        print(f'{x = }, {x1 = }, {y = }, {y1 = }, ')
        cv2.arrowedLine(frame, (x, y), (x1, y1), (0, 255, 0), 1, tipLength=0.3)
    return frame


video = cv2.VideoCapture("Counter-Strike 2 - 2024-12-20 11-49-05.mp4")
ret, prev_frame = video.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

output_file = "output.mp4"
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
frame_height, frame_width, = prev_gray.shape[:2]
fps = 60
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = harris_corner_detector(gray)

    flow = lucas_kanade(prev_gray, gray, corners)

    output_frame = draw_optical_flow(frame, flow)
    out.write(output_frame)
    cv2.imshow("optical flow", output_frame)
    prev_gray = gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
out.release()
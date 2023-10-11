import numpy as np
from cv2 import cv2


def calculate_dissimilarity(m1, m2):
    # assert m1.shape == m2.shape, "Maps must have the same shape"
    print(m1)
    print(m2)
    rows, cols, chanel = m1.shape
    dissimilarity = 0

    print("开始")
    for c in [0, 1]:  # Considering only two grid states: obstacle (1) and non-obstacle (0)
        c_m1 = np.where(m1 == c)  # Get indices of grid cells with state c in m1
        c_m2 = np.where(m2 == c)  # Get indices of grid cells with state c in m2

        for p1 in zip(c_m1[0], c_m1[1]):
            min_dist = np.inf

            for p2 in zip(c_m2[0], c_m2[1]):
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])  # Manhattan distance
                min_dist = min(min_dist, dist)

            dissimilarity += min_dist / len(c_m1[0])  # Normalize by the number of grid cells with state c in m1
    print("结束")
    return -dissimilarity


if __name__ == '__main__':

    m1 = cv2.imread("img/merge1_img1.png")
    m2 = cv2.imread("img/merge1_img1.png")
    # 将图片转换为矩阵
    # m11 = m1 > 127 1:0
    m1 = np.where(m1 > 127, 1, 0)
    m2 = np.where(m2 > 127, 1, 0)
    print(m1.shape)
    print(m2.shape)
    print(m1 == m2)
    # Example usage
    # m1 = np.array([[0, 1, 0],
    #                [1, 0, 1],
    #                [0, 0, 1]])
    #
    # m2 = np.array([[0, 0, 1, 1],
    #                [1, 0, 0, 0],
    #                [0, 0, 1, 1]])

    dissimilarity = calculate_dissimilarity(m1, m2)
    print("Dissimilarity:", dissimilarity)

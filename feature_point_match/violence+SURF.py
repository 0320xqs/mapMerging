import numpy as np
import cv2
from matplotlib import pyplot as plt


def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))

    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


def ByBFMatcher(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
    """
    （1）暴力法
    :param img1: 匹配图像1
    :param img2: 匹配图像2
    :param kp1: 匹配图像1的特征点
    :param kp2: 匹配图像2的特征点
    :param des1: 匹配图像1的描述子
    :param des2: 匹配图像2的描述子
    :return:
    """
    if flag == "SIFT" or flag == "sift":
        # SIFT方法或SURF
        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
    else:
        # ORB方法
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    ms = bf.match(des1, des2)
    # ms = sorted(ms, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("1.jpg", img3)
    cv2.imshow("Matches", img3)
    cv2.waitKey(0)
    return ms


if __name__ == '__main__':
    img1 = cv2.imread("asset/image/demo1.jpg")
    img2 = cv2.imread("asset/image/demo1.jpg")
    # img1 = cv2.imread("asset/image/part01-2.jpg")
    # img2 = cv2.imread("asset/image/part02-2.jpg")
    kp1, des1 = SURF(img1)
    kp2, des2 = SURF(img2)
    matches = ByBFMatcher(img1, img2, kp1, kp2, des1, des2, "SIFT")

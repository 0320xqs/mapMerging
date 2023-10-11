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

def ORB(img):
    """
     ORB角点检测
     实例化ORB对象
    """
    orb = cv2.ORB_create(nfeatures=500)
    """检测关键点，计算特征描述符"""
    kp, des = orb.detectAndCompute(img, None)

    # 将关键点绘制在图像上
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    cv2.imwrite("1.jpg", img2)

    # 画图
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des

def ByFlann(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
    """
        （1）FLANN匹配器
        :param img1: 匹配图像1
        :param img2: 匹配图像2
        :param kp1: 匹配图像1的特征点
        :param kp2: 匹配图像2的特征点
        :param des1: 匹配图像1的描述子
        :param des2: 匹配图像2的描述子
        :return:
        """
    if (flag == "SIFT" or flag == "sift"):
        # SIFT方法
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_params = dict(check=50)
    else:
        # ORB方法
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(check=50)
    # 定义FLANN参数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", img3)
    cv2.imwrite("1.jpg", img3)
    cv2.waitKey(0)
    return matches


if __name__ == '__main__':
    img1 = cv2.imread("asset/image/part01-2.jpg")
    img2 = cv2.imread("asset/image/part02-2.jpg")
    kp1, des1 = ORB(img1)
    kp2, des2 = ORB(img2)
    matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "ORB")

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
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matches", img3)
    # cv2.imwrite("1.jpg", img3)
    # cv2.waitKey(0)
    return matches


def RANSAC(img1, img2, kp1, kp2, matches):
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    matchType = type(matches[0])
    good = []
    print(matchType)
    if isinstance(matches[0], cv2.DMatch):
        # 搜索使用的是match
        good = matches
    else:
        # 搜索使用的是knnMatch
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    draw_params1 = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=None,  # draw only inliers
                        flags=2)

    img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)
    cv2.imwrite("1.jpg", img33)
    cv2.imwrite("2.jpg", img3)
    cv2.imshow("before", img33)
    cv2.imshow("now", img3)
    cv2.waitKey(0)


def SIFT(img):
    # SIFT算法关键点检测
    # 读取图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT关键点检测
    # 1. 实例化sift
    sift = cv2.xfeatures2d.SIFT_create()

    # 2. 利用sift.detectAndCompute()检测关键点并计算
    kp, des = sift.detectAndCompute(gray, None)
    # gray: 进行关键带你检测的图像，注意是灰度图像
    # kp: 关键点信息，包括位置，尺度，方向信息
    # des: 关键点描述符，每个关键点对应128个梯度信息的特征向量

    # 3. 将关键点检测结果绘制在图像上
    # cv2.drawKeypoints(image, keypoints, outputimage, color, flags)
    # image: 原始图像
    # keypoints: 关键点信息，将其绘制在图像上
    # outputimage: 输出图片，可以是原始图像
    # color: 颜色设置，通过修改(b, g, r)的值，更改画笔的颜色，b = 蓝色, g = 绿色, r = 红色
    # flags: 绘图功能的标识设置
    # 1. cv2.DRAW_MATCHES_FLAGS_DEFAULT: 创建输出图像矩阵，使用现存的输出图像绘制匹配对象和特征点，对每一个关键点只绘制中间点
    # 2. cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: 不创建输出图像矩阵，而是在输出图像上绘制匹配对
    # 3. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 对每一个特征点绘制带大小和方向的关键点图形
    # 4. cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: 单点的特征点不被绘制
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))
    cv2.imwrite("1.jpg", img)
    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


if __name__ == '__main__':
    img1 = cv2.imread("asset/image/part01-2.jpg")
    img2 = cv2.imread("asset/image/part02-2.jpg")
    # img1 = cv2.imread("asset/image/part01-2.jpg")
    # img2 = cv2.imread("asset/image/part02-2.jpg")
    kp1, des1 = SIFT(img1)
    kp2, des2 = SIFT(img2)
    matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "SIFT")
    RANSAC(img1, img2, kp1, kp2, matches)

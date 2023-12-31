"""
图像特征点的检测与匹配
主要涉及：
1、ORB
2、SIFT
3、SURF
"""

"""
一、图像特征点的检测
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


def ORB(img):
    """
     ORB角点检测
     实例化ORB对象
    """
    orb = cv2.ORB_create(nfeatures=500)
    """检测关键点，计算特征描述符"""
    kp, des = orb.detectAndCompute(img, None)

    # 将关键点绘制在图像上
    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)

    # 画图
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


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
    cv2.drawKeypoints(img, kp, None, (0, 255, 0))

    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kp, None, (0, 255, 0))

    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


"""
2.图像特征点匹配方法
（1）暴力法
（2）FLANN匹配器
"""


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
    if (flag == "SIFT" or flag == "sift"):
        # SIFT方法
        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
    else:
        # ORB方法
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    ms = bf.knnMatch(des1, des2, k=2)
    # ms = sorted(ms, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matches", img3)
    # cv2.waitKey(0)
    return ms


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
    # matches = flann.match(des1, des2)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


"""
优化匹配结果
RANSAC算法是RANdom SAmple Consensus的缩写,意为随机抽样一致
"""


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
        for match in matches:
            # print(len(match))
            if len(match) > 1:
                m = match[0]
                n = match[1]
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            else:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # M, mask = cv2.estimateAffine2D(src_pts, dst_pts)
        # 运行 RANSAC 结合最小二乘法估计仿射矩阵
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        M = np.vstack((M, [0, 0, 1]))  # 把2*3变成3*3
        matchesMask = mask.ravel().tolist()
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

    show("NO_RANSAC", img33)
    show("RANSAC", img3)
    return img1, img2, M


"""
融合图片结果
"""


def merge(img1, img2, M):
    # 计算融合后的图像ROI
    height, width, m = ROI(img1, img2, M)

    M[0, 2] += m[0, 2]  # 设置水平平移分量
    M[1, 2] += m[1, 2]  # 设置垂直平移分量
    # 对第一张图片进行透视变换
    transform_img1 = cv2.warpPerspective(img1, M, (width, height), borderValue=(125, 125, 125))
    show("Transform_img1", transform_img1)
    cv2.imwrite("asset/result/stillness_right1.jpg", transform_img1)
    print("变换矩阵为：\n%s" % M)

    print("平移矩阵为：\n%s" % m)

    # 对齐img2左上角和融合后的左上角
    # 用仿射变换实现平移,
    img2_transform = cv2.warpAffine(img2, m, (width, height), borderValue=(125, 125, 125))
    show("img2_transform", img2_transform)
    cv2.imwrite("asset/result/stillness_right2.jpg", img2_transform)

    # 将两张图片并排在一起显示
    # connect_image = np.concatenate((transform_img1, img2), axis=1)
    # cv2.imshow('connect_image', connect_image)
    # cv2.waitKey(0)

    # # 将图像叠加在一起
    merged_image = cv2.addWeighted(transform_img1, 1, img2_transform, 0.1, 0)

    # 按照自定义方式组合图像像素
    # merged_image = np.zeros_like(transform_img1)  # 创建一个与 img1 相同形状的空图像
    # 根据自定义规则对像素进行组合
    # for i in range(height):
    #     for j in range(width):
    #         merged_image[i, j] = np.maximum(transform_img1[i, j], img2_transform[i, j])
    #         # 黑色都显示出来
    #         if np.all(transform_img1[i, j] == 0) or np.all(img2_transform[i, j] == 0):
    #             merged_image[i, j] = 0

    show("merged_image", merged_image)


"""
计算ROI边框
"""


def ROI(img1, img2, M):
    # 计算 img1 的边界角点
    height, width = img1.shape[:2]
    print(height)
    print(width)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    print("原始角点：\n %s" % corners)
    # 获取变换后的角点坐标，变换前需要对数据进行整理（变成float类型，同时二维变成三维，一个都不能少）
    transformed_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M)
    # 再变回二维
    transformed_corners = transformed_corners.squeeze().astype(np.int32)
    # return transformed_corners

    print("变换后角点为：\n %s" % transformed_corners)
    # 计算最左
    merged_left = min(transformed_corners[:, 0].min(), 0)
    # 计算最右
    merged_right = max(transformed_corners[:, 0].max(), img2.shape[1])
    # 计算最上
    merged_top = min(transformed_corners[:, 1].min(), 0)
    # 计算最下
    merged_bottom = max(transformed_corners[:, 1].max(), img2.shape[0])
    # 计算融合后图像的宽度和高度
    merged_width = merged_right - merged_left
    merged_height = merged_bottom - merged_top
    print("最左面：%s" % merged_left)
    print("最右面：%s" % merged_right)
    print("最上面：%s" % merged_top)
    print("最下面：%s" % merged_bottom)
    print("宽度：%s" % merged_width)
    print("高度：%s" % merged_height)
    # 定义平移矩阵，需要是numpy的float32类型 x轴平移-左点，y轴平移-右点（让左上角归零）
    m = np.float32([[1, 0, -merged_left], [0, 1, -merged_top]])
    return merged_height, merged_width, m


def show(name, img_show):
    img_show_resize = cv2.resize(img_show, (0, 0), fx=img_resize, fy=img_resize)
    # cv2.namedWindow(name, 0)
    cv2.imshow(name, img_show_resize)
    # cv2.waitKey(0)
    # cv2.namedWindow(name, 0)
    # cv2.resizeWindow(name,20,20)
    # cv2.imshow(name, img_show)


img_resize = 0.5
if __name__ == '__main__':
    img1 = cv2.imread("asset/image/local_06_1.pgm")
    img2 = cv2.imread("asset/image/local_06_2.pgm")
    print(img1.shape[:])
    print(img2.shape[:])
    # 提取特征点
    kp1, des1 = SIFT(img1)
    kp2, des2 = SIFT(img2)
    # 特征点匹配
    matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "SIFT")
    # RANSAC优化，获取Transform
    img1, img2, M = RANSAC(img1, img2, kp1, kp2, matches)

    show("img1", img1)
    show("img2", img2)
    # cv2.waitKey(0)

    # 图片融合
    merge(img1, img2, M)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("融合结束！")

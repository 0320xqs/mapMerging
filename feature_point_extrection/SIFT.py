import numpy as np
import cv2
from matplotlib import pyplot as plt


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
    kp1, des1 = SIFT(img1)

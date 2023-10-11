import numpy as np
import cv2
from matplotlib import pyplot as plt


def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()  # 创建出surf对象
    kp, des = surf.detectAndCompute(img, None)  # 检测特征点and计算描述子
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))  # 将特征点描绘出来

    cv2.imwrite("1.jpg", img)  # 保存绘制的图片
    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)  # 指定大小和分辨率的图形窗口
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴的刻度线和刻度标签
    plt.show()
    return kp, des


if __name__ == '__main__':
    img1 = cv2.imread("asset/image/part02-2.jpg")
    kp1, des1 = SURF(img1)

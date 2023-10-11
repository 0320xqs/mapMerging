import cv2
import numpy as np

import math


def line_detect(image):
    # 将图片转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 设置阈值
    lowera = np.array([0, 0, 221])
    uppera = np.array([180, 30, 255])
    mask1 = cv2.inRange(hsv, lowera, uppera)
    kernel = np.ones((3, 3), np.uint8)

    # 对得到的图像进行形态学操作（闭运算和开运算）
    mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)  # 闭运算：表示先进行膨胀操作，再进行腐蚀操作
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算：表示的是先进行腐蚀，再进行膨胀操作

    # 绘制轮廓
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    # 显示图片
    cv2.imshow("edges", edges)
    # 检测白线    这里是设置检测直线的条件，可以去读一读HoughLinesP()函数，然后根据自己的要求设置检测条件
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 12, minLineLength=25, maxLineGap=2)
    # 在图片上画直线
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            image_line = cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 2)
    # 显示图片
    cv2.imshow("img_line", image_line)
    print("lines=", lines)
    print("========================================================")
    print("直线数量为：",len(lines))
    cv2.waitKey(0)
    return lines


if __name__ == '__main__':
    # 读入图片
    # src = cv2.imread("img/merge2_img2.jpg")
    src = cv2.imread("img/merge1_img1.png")
    # 设置窗口大小
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    # 显示原始图片
    cv2.imshow("input image", src)

    # 调用评价函数
    lines = line_detect(src)
    print("分数为：",len(lines))
    cv2.waitKey(0)


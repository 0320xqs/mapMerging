from copy import deepcopy

import cv2
import numpy as np

import math


# 判断欧式距离
def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


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
    print("lines=", lines)
    print("========================================================")
    i = 1
    # 对通过霍夫变换得到的数据进行遍历
    for line in lines:
        # newlines1 = lines[:, 0, :]
        print("line[" + str(i - 1) + "]=", line)
        x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        print(line[0])
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 在原图上画线
        # 转换为浮点数，计算斜率
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        if x2 - x1 == 0:
            print("直线是竖直的")
            result = 90
        elif y2 - y1 == 0:
            print("直线是水平的")
            result = 0
        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
            print("直线倾斜角度为：%s 度" % str(result))
        i = i + 1
    #     显示最后的成果图
    cv2.imshow("line_detect", image)
    return lines


def group_lines(lines, image):
    groups = []  # 存储直线组的列表
    compared = set()  # 存储已经比较过的直线的集合

    for i in range(len(lines)):
        if i in compared:
            continue

       # 创建一个包含直线1的新组
        for j in range(i + 1, len(lines)):
            if j in compared:
                continue
            line1 = lines[i]
            x1, y1, x2, y2 = line1[0]  # 获取直线1的起点和终点坐标
            group = [line1]

            line2 = lines[j]
            x3, y3, x4, y4 = line2[0]  # 获取直线2的起点和终点坐标

            # 计算直线1和直线2的起点或终点之间的欧氏距离
            # distance = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)

            # 判断距离是否小于2，若是则将直线2加入当前组
            # if distance < 2:
            if (euclidean_distance(x1, y1, x3, y3) <= 10 or euclidean_distance(x1, y1, x4, y4) <= 10 or
                    euclidean_distance(x2, y2, x3, y3) <= 10 or euclidean_distance(x2, y2, x4, y4) <= 10):
                group.append(line2)
                compared.add(j)
            if (len(group) == 2):
                groups.append(group)
        # 输出每个直线组的直线列表
    print("总共有：")
    print(len(groups))
    for i, group in enumerate(groups):
        print(f"Group {i + 1} lines:")
        list = []
        for line in group:
            print(line[0])
            list.append(line[0])
        print()
        calculate_angle(list[0], list[1], image)
    return groups


def calculate_angle(line1, line2, image):
    x1, y1, x2, y2 = line1  # 直线1的起点和终点坐标
    x3, y3, x4, y4 = line2  # 直线2的起点和终点坐标
    print("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
    # 计算直线1的向量表示
    v1 = (x2 - x1, y2 - y1)
    # 计算直线2的向量表示
    v2 = (x4 - x3, y4 - y3)

    # 计算直线1和直线2的夹角（弧度）
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    # 检查参数的有效性
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("One or both vectors have zero length.")

    if dot_product / (norm_v1 * norm_v2) > 1 or dot_product / (norm_v1 * norm_v2) < -1:
        raise ValueError("Invalid value for the acos function.")
    angle_rad = math.acos(dot_product / (norm_v1 * norm_v2))
    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)
    print(angle_deg)
    if (angle_deg == 90.0):
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
    else:
        print("33333")
        #     显示最后的成果图
    cv2.imshow("RTangle_detect", image)
    return angle_deg


if __name__ == '__main__':
    # 读入图片
    src = cv2.imread("right-angle-detect.pgm")
    # 设置窗口大小
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    # 显示原始图片
    cv2.imshow("input image", src)
    # 调用函数
    lines = line_detect(src)

    group_lines(lines, src)
    cv2.waitKey(0)

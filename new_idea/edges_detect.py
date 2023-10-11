import cv2
import numpy as np


def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4, tolerance):
    # 计算直线方程参数
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2 * y1 - x1 * y2

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x4 * y3 - x3 * y4

    # 计算交点的坐标
    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < 1e-6:
        # 直线平行或重合，无交点
        return None

    intersection_x = int(abs((c1 * b2 - c2 * b1) / determinant))
    intersection_y = int(abs((a1 * c2 - a2 * c1) / determinant))

    # 计算两条线段是否与圆相交
    dist1 = np.sqrt((intersection_x - x1) ** 2 + (intersection_y - y1) ** 2)
    dist2 = np.sqrt((intersection_x - x2) ** 2 + (intersection_y - y2) ** 2)
    dist3 = np.sqrt((intersection_x - x3) ** 2 + (intersection_y - y3) ** 2)
    dist4 = np.sqrt((intersection_x - x4) ** 2 + (intersection_y - y4) ** 2)

    if min(dist1, dist2) <= tolerance and min(dist3, dist4) <= tolerance:
        return intersection_x, intersection_y
    else:
        return None


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def detect_and_highlight_right_angles(input_image_path, output_image_path):
    # 1. 读取pgm格式的地图图片
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 2. 使用Canny边缘检测算法提取边缘并用绿色突出显示
    edges = cv2.Canny(input_image, 30, 150)
    # 检查是否成功提取到边缘
    if np.sum(edges) == 0:
        print("未能成功提取到边缘，请检查图像或调整阈值。")
        return
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[edges != 0] = [0, 255, 0]  # 将边缘的颜色设置为绿色
    cv2.imshow('Canny Edges:GREEN', edges_colored)
    # cv2.waitKey(0)

    # 3. 对提取的边缘执行RANSAC直线段检测并用蓝色突出显示
    lines = cv2.HoughLinesP(edges, 1, np.pi / 90, threshold=13, minLineLength=13, maxLineGap=4)
    output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    print("nums of Lines:", len(lines))

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 用蓝色绘制直线
        cv2.imshow('RANSAC Lines:BLUE', output_image)
        # cv2.waitKey(0)

        # 4. 对检测的直线段进行判断并用红色突出显示
        output_final = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        other = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        right_angle_detected = False
        tolerance = 20
        rightanglenum = 0
        vertical_flag = np.zeros(len(lines), dtype=int)

        if lines is not None and len(lines) >= 2:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    angle_diff = abs(
                        np.degrees(np.arctan2(y2 - y1, x2 - x1)) - np.degrees(np.arctan2(y4 - y3, x4 - x3)))

                    denominator = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4, tolerance)
                    if denominator:
                        intersection_x, intersection_y = denominator
                        print("intersection_x,intersection_y:", intersection_x, intersection_y)
                        print("两条线段的端点分别为：", x1, y1, x2, y2, x3, y3, x4, y4)
                    else:
                        continue

                    if 85 <= angle_diff <= 95:
                        if vertical_flag[i] != 1 and vertical_flag[j] != 1:
                            vertical_flag[i] = 1
                            vertical_flag[j] = 1

                            cv2.line(other, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cv2.line(other, (x3, y3), (x4, y4), (255, 0, 0), 1)
                            print("两条垂直线段的端点分别为：", x1, y1, x2, y2, x3, y3, x4, y4)

                            if distance(intersection_x, intersection_y, x1, y1) >= distance(intersection_x,
                                                                                            intersection_y,
                                                                                            x2, y2):
                                cv2.line(output_final, (intersection_x, intersection_y), (x1, y1), (0, 0, 255),
                                         1)  # 用红色绘制第一条直线
                            else:
                                cv2.line(output_final, (intersection_x, intersection_y), (x2, y2), (0, 0, 255),
                                         1)  # 用红色绘制第一条直线

                            if distance(intersection_x, intersection_y, x3, y3) >= distance(intersection_x,
                                                                                            intersection_y,
                                                                                            x4, y4):
                                cv2.line(output_final, (intersection_x, intersection_y), (x3, y3), (0, 0, 255),
                                         1)  # 用红色绘制第一条直线
                            else:
                                cv2.line(output_final, (intersection_x, intersection_y), (x4, y4), (0, 0, 255),
                                         1)  # 用红色绘制第一条直线

                            right_angle_detected = True
                            rightanglenum += 1
                        else:
                            if vertical_flag[i] == 1:
                                vertical_flag[j] = 1
                            else:
                                if vertical_flag[j] == 1:
                                    vertical_flag[i] = 1
                    else:
                        print("两条直线段不垂直！")

        # # 保存为png格式图片
        if right_angle_detected:
            # cv2.imwrite('map/output.png', output_final)
            # print(f"直角墙线已经突出并保存到 {output_image_path}")

            # 展示最终结果
            print("num of rightAngle:", rightanglenum)
            cv2.imshow('Final Result', output_final)
            cv2.imshow('other:', other)
            cv2.waitKey(0)
        else:
            print("未检测到直角墙线。")
    else:
        print("未检测到直线")


def main():
    input_image_path = 'map/part02-2.pgm'  # 输入图像的路径
    output_image_path = 'map/output_image.png'  # 输出图像的路径

    detect_and_highlight_right_angles(input_image_path, output_image_path)


if __name__ == "__main__":
    main()

# 显示
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as mpl_polygon
import matplotlib.pyplot as plt


# 计算变换后的矩形的四个角点
def transform_corners(img, M):
    # 计算 img1 的边界角点
    height, width = img.shape[:2]

    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    print("原始角点：\n %s" % corners)
    # 获取变换后的角点坐标，变换前需要对数据进行整理（变成float类型，同时二维变成三维，一个都不能少）
    transformed_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M)
    # 再变回二维
    transformed_corners = transformed_corners.squeeze().astype(np.int32)
    print("变换后角点为：\n %s" % transformed_corners)
    return transformed_corners


# 重叠区域处理
def overlap(img1_before, img2_before, img1_transform, img2_transform, M, m):
    img1_transform_corners = transform_corners(img1_before, M)
    img2_transform_corners = transform_corners(img2_before, m)
    img1_rect = [tuple(point) for point in img1_transform_corners]
    img2_rect = [tuple(point) for point in img2_transform_corners]
    print(img1_rect)
    print(img2_rect)
    # 创建矩形1的Polygon对象
    poly1 = Polygon(img1_rect)
    # 创建矩形2的Polygon对象
    poly2 = Polygon(img2_rect)

    # 检查两个矩形是否有重叠区域
    if poly1.intersects(poly2):
        # 计算重叠区域的Polygon对象
        overlap = poly1.intersection(poly2)
        # 提取重叠区域的坐标信息
        overlap_coords = list(overlap.exterior.coords)
        print("重合部分角点:")
        print(overlap_coords)
        # 画出来两个矩形及其重叠区域
        draw_overlap(img1_rect, img2_rect, overlap_coords)
        print(overlap.bounds[0])
        print(overlap.bounds[1])
        print(overlap.bounds[2])
        print(overlap.bounds[3])

        compute_sum = 0
        overlap_sum = 0
        effective_sum = 0
        alignment_sum = 0
        # 遍历重叠区域内的像素点
        sample = 1  # 抽样指标
        for x in range(int(overlap.bounds[0]), int(overlap.bounds[2]) + 1, sample):
            for y in range(int(overlap.bounds[1]), int(overlap.bounds[3]) + 1, sample):
                compute_sum += 1
                if overlap.contains(Point(x, y)):
                    overlap_sum += 1
                    # print("像素点 ({}, {}) 在重叠区域内".format(x, y))
                    # print(len(img1_transform[1]))
                    # print(len(img1_transform))
                    # print(len(img2_transform[1]))
                    # print(len(img2_transform))
                    # 计算对齐程度
                    if (img1_transform[y][x] == 255 or img1_transform[y][x] == 0) and (
                            img2_transform[y][x] == 255 or img2_transform[y][x] == 0):
                        effective_sum += 1
                        if img1_transform[y][x] == img2_transform[y][x]:
                            alignment_sum += 1
                else:
                    pass
                    # print("像素点 ({}, {}) 不在重叠区域内".format(x, y))
        print(f"重叠部分计算（步长为：{sample}）的数量：{compute_sum}")
        print(f"实际重叠部分（步长为：{sample}）的数量：{overlap_sum}")
        print(f"有效重叠部分（步长为：{sample}）的数量：{effective_sum}")
        print(f"对齐重叠部分（步长为：{sample}）的数量：{alignment_sum}")
        print(f"对齐/有效（（步长为：{sample}））: {alignment_sum / effective_sum}")
    else:
        # 如果没有重叠区域，则返回空值
        print("如果没有重叠区域")


# 画出轮廓
def draw_overlap(img1_rect, img2_rect, overlap_coords):
    # 绘制两个矩形
    fig, ax = plt.subplots()
    rectangle1_patch = mpl_polygon(img1_rect, ec='blue', fc='none')
    rectangle2_patch = mpl_polygon(img2_rect, ec='red', fc='none')
    ax.add_patch(rectangle1_patch)
    ax.add_patch(rectangle2_patch)

    # 绘制相交部分
    intersection_patch = mpl_polygon(overlap_coords, ec='green', fc='none')
    ax.add_patch(intersection_patch)

    # 设置坐标轴范围
    ax.set_xlim([0, img1_transform.shape[1]])
    ax.set_ylim([img1_transform.shape[0], 0])
    # 显示图形
    plt.show()


# 输出显示图片
def show(name, img_show):
    img_show_resize = cv2.resize(img_show, (0, 0), fx=img_resize, fy=img_resize)
    cv2.imshow(name, img_show_resize)


img_resize = 0.5
if __name__ == '__main__':
    img1_transform = cv2.imread("img/img1_transform.jpg", cv2.IMREAD_GRAYSCALE)
    img2_transform = cv2.imread("img/img2_transform.jpg", cv2.IMREAD_GRAYSCALE)
    img1_before = cv2.imread("img/img1_before.pgm", cv2.IMREAD_GRAYSCALE)
    img2_before = cv2.imread("img/img2_before.pgm", cv2.IMREAD_GRAYSCALE)
    M = np.array([[-7.66168198e-01, -6.39212573e-01, 1.15231861e+03],
                  [6.39212573e-01, -7.66168198e-01, 6.27826540e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    m = np.array([[1., 0., 262.],
                  [0., 1., 127.],
                  [0., 0., 1.]])
    # img1_transform = cv2.imread("img/stillness_right1.jpg", cv2.IMREAD_GRAYSCALE)
    # img2_transform = cv2.imread("img/stillness_right2.jpg", cv2.IMREAD_GRAYSCALE)
    # img1_before = cv2.imread("img/stillness1.jpg", cv2.IMREAD_GRAYSCALE)
    # img2_before = cv2.imread("img/stillness2.jpg", cv2.IMREAD_GRAYSCALE)
    # M = np.array([[1.00027183e+00, 5.83500428e-05, -1.88623363e-01],
    #               [-5.83500428e-05, 1.00027183e+00, -1.10935165e-02],
    #               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # m = np.array([[1., 0., 0.],
    #               [0., 1., 0.],
    #               [0., 0., 1.]])
    # M = np.array([[0.93208005, 0.38811401, - 0.78896301],
    #               [-0.38811401, 0.93208005, 310.06306853],
    #               [0., 0., 1.]])
    # m = np.array([[1., 0., 38.],
    #               [0., 1., 43.],
    #               [0., 0., 1.]])
    show("img1", img1_transform)
    show("img2", img2_transform)
    show("img1_before", img1_before)
    show("img2_before", img2_before)
    cv2.waitKey(0)

    print("转换后宽: " + str(img1_transform.shape[1]))
    print("转换后高: " + str(img1_transform.shape[0]))
    print("转换后宽: " + str(img1_before.shape[1]))
    print("转换后高: " + str(img1_before.shape[0]))
    print(M)
    print(m)
    overlap(img1_before, img2_before, img1_transform, img2_transform, M, m)
    # distance = evaluate_similarity(img1, img2)
    # print("距离为: " + str(distance))
    cv2.destroyAllWindows()

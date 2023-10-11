import cv2
import numpy as np


# 显示
def show(name, img_show):
    img_show_resize = cv2.resize(img_show, (0, 0), fx=img_resize, fy=img_resize)
    cv2.imshow(name, img_show_resize)


# 计算d_mapc(算是一种优化，没有这个操作，就需要迭代m1*n1次m2*n2)
def compute_dmapc(c, matric):
    max_x = matric.shape[1]
    max_y = matric.shape[0]
    print(max_x)
    print(max_y)
    d_map_c = np.empty((max_x, max_y))
    # 初始化
    for y in range(max_y):
        for x in range(max_x):
            if (matric[y][x] == c):
                d_map_c[x][y] = 0
            else:
                d_map_c[x][y] = max_x + max_y + 1
    # 从左上到右下
    for y in range(1, max_y - 1):
        for x in range(1, max_x - 1):
            h = min(d_map_c[x - 1][y] + 1, d_map_c[x][y - 1] + 1)
            d_map_c[x][y] = min(d_map_c[x][y], h)

    # 从右下到左上
    for y in range(max_y - 2, 0, -1):
        for x in range(max_x - 2, 0, -1):
            h = min(d_map_c[x + 1][y] + 1, d_map_c[x][y + 1] + 1)
            d_map_c[x][y] = min(d_map_c[x][y], h)
    return d_map_c


# 计算总距离
def evaluate_similarity(img1, img2):
    assert img1.shape[0] == img2.shape[0], "两张图片高度不同"
    assert img1.shape[1] == img2.shape[1], "两张图片宽度不同"
    max_x = img1.shape[1]
    max_y = img1.shape[0]
    c_list = [0, 255]
    d_total = 0
    for c in c_list:
        # 计算d(m1,m2，c)
        d_map_c = compute_dmapc(c, img2)
        d1_c = 0
        d1_c_sum = 0
        for y in range(max_y):
            for x in range(max_x):
                if (img1[y][x] == c):
                    d1_c = d1_c + d_map_c[x][y]
                    d1_c_sum += 1

        # 计算d(m2,m1，c)
        d_map_c = compute_dmapc(c, img1)
        d2_c = 0
        d2_c_sum = 0
        for y in range(max_y):
            for x in range(max_x):
                if (img2[y][x] == c):
                    d2_c = d2_c + d_map_c[x][y]
                    d2_c_sum += 1

        d_total = d_total + d1_c / d1_c_sum + d2_c / d2_c_sum

    return d_total


img_resize = 0.5
if __name__ == '__main__':
    img1 = cv2.imread("img/img1_transform.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("img/img2_transform.jpg", cv2.IMREAD_GRAYSCALE)
    show("img1", img1)
    show("img2", img2)
    cv2.waitKey(0)

    print("宽: " + str(img1.shape[1]))
    print("高: " + str(img1.shape[0]))
    distance = evaluate_similarity(img1, img2)
    print("距离为: " + str(distance))
    cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import cos, pi, sin
from 计算刻度值 import get_rad_val

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = cv2.TM_CCOEFF


def get_match_rect(template, img, method):
    '''获取模板匹配的矩形的左上角和右下角的坐标'''
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(img, template, method)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的方法，对结果的解释不同
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right


def get_center_point(top_left, bottom_right):
    '''传入左上角和右下角坐标，获取中心点'''
    c_x, c_y = ((np.array(top_left) + np.array(bottom_right)) / 2).astype(np.int)
    return c_x, c_y


def get_circle_field_color(img, center, r, thickness):
    '''获取中心圆形区域的色值集'''
    temp = img.copy().astype(np.int)
    cv2.circle(temp, center, r, -100, thickness=thickness)
    return img[temp == -100]


def v2_by_center_circle(img, colors):
    '''二值化通过中心圆的颜色集合'''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = img[i, j]
            if a in colors:
                img[i, j] = 0
            else:
                img[i, j] = 255


def v2_by_k_means(img):
    '''使用k-means二值化'''
    original_img = np.array(img, dtype=np.float64)
    src = original_img.copy()
    delta_y = int(original_img.shape[0] * (0.4))
    delta_x = int(original_img.shape[1] * (0.4))
    original_img = original_img[delta_y:-delta_y, delta_x:-delta_x]
    h, w, d = src.shape
    print(w, h, d)
    dts = min([w, h])
    print(dts)
    r2 = (dts / 2) ** 2
    c_x, c_y = w / 2, h / 2
    a: np.ndarray = original_img[:, :, 0:3].astype(np.uint8)
    # 获取尺寸(宽度、长度、深度)
    height, width = original_img.shape[0], original_img.shape[1]
    depth = 3
    print(depth)
    image_flattened = np.reshape(original_img, (width * height, depth))
    '''
    用K-Means算法在颜色样本中建立2个类。
    '''
    image_array_sample = shuffle(image_flattened, random_state=0)
    estimator = KMeans(n_clusters=2, random_state=0)
    estimator.fit(image_array_sample)
    '''
    我们为原始图片的每个像素进行类的分配。
    '''
    src_shape = src.shape
    new_img_flattened = np.reshape(src, (src_shape[0] * src_shape[1], depth))
    cluster_assignments = estimator.predict(new_img_flattened)
    compressed_palette = estimator.cluster_centers_
    print(compressed_palette)
    a = np.apply_along_axis(func1d=lambda x: np.uint8(compressed_palette[x]), arr=cluster_assignments, axis=0)
    img = a.reshape(src_shape[0], src_shape[1], depth)
    print(compressed_palette[0, 0])
    threshold = (compressed_palette[0, 0] + compressed_palette[1, 0]) / 2
    img[img[:, :, 0] > threshold] = 255
    img[img[:, :, 0] < threshold] = 0
    cv2.imshow('sd0', img)
    for x in range(w):
        for y in range(h):
            distance = ((x - c_x) ** 2 + (y - c_y) ** 2)
            if distance > r2:
                pass
                img[y, x] = (255, 255, 255)
    cv2.imshow('sd', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def get_pointer_rad(img):
    '''获取角度'''
    shape = img.shape
    c_y, c_x, depth = int(shape[0] / 2), int(shape[1] / 2), shape[2]
    x1 = c_x + c_x * 0.8
    src = img.copy()
    freq_list = []
    for i in range(361):
        x = (x1 - c_x) * cos(i * pi / 180) + c_x
        y = (x1 - c_x) * sin(i * pi / 180) + c_y
        temp = src.copy()
        cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=3)
        t1 = img.copy()
        t1[temp[:, :, 2] == 255] = 255
        c = img[temp[:, :, 2] == 255]
        points = c[c == 0]
        freq_list.append((len(points), i))
        cv2.imshow('d', temp)
        cv2.imshow('d1', t1)
        cv2.waitKey(1)
    print('当前角度：', max(freq_list, key=lambda x: x[0]), '度')
    cv2.destroyAllWindows()
    return max(freq_list, key=lambda x: x[0])


if __name__ == '__main__':
    for x in range(1, 32):
        # 获取测试图像
        img_s = cv2.imread('test/ (%s).jpg' % x)
        img = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('template1.png')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # 匹配并返回矩形坐标
        top_left, bottom_right = get_match_rect(template, img, method)
        c_x, c_y = get_center_point(top_left, bottom_right)
        print(c_x, c_y)
        # 绘制矩形
        cv2.rectangle(img_s, top_left, bottom_right, 255, 2)
        cv2.imshow('img', cv2.resize(img_s, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #################################################################
        new = img_s[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        template = cv2.imread('template.png')
        top_left, bottom_right = get_match_rect(template, new, method=method)
        new_ = new[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # 二值化图像
        cv2.imshow('ererss', new_)
        img = v2_by_k_means(new_)
        rad = get_pointer_rad(img)
        #################################################################
        print(get_rad_val(rad[1]), '对应刻度')
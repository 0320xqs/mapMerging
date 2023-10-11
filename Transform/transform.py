import cv2
import numpy as np


def rotation(img, img_rotation):
    h, w, c = img.shape
    # # #######获取没有裁剪的旋转图像#########
    # 定义空矩阵
    M = np.zeros((2, 3), dtype=np.float32)
    # 设定旋转角度
    alpha = np.cos((img_rotation / 180) * np.pi)
    beta = np.sin((img_rotation / 180) * np.pi)
    print('alpha: ', alpha)
    # 初始化旋转矩阵
    M[0, 0] = alpha
    M[1, 1] = alpha
    M[0, 1] = beta
    M[1, 0] = -beta
    # 图片中心点坐标
    cx = w / 2
    cy = h / 2

    # 变化的宽高
    tx = (1 - alpha) * cx - beta * cy
    ty = beta * cx + (1 - alpha) * cy
    M[0, 2] = tx
    M[1, 2] = ty

    # 旋转后的图像高、宽
    rotated_w = int(h * np.abs(beta) + w * np.abs(alpha))
    rotated_h = int(h * np.abs(alpha) + w * np.abs(beta))

    # 移动后的中心位置
    M[0, 2] += rotated_w / 2 - cx
    M[1, 2] += rotated_h / 2 - cy

    result = cv2.warpAffine(img, M, (rotated_w, rotated_h))

    # 图片修正
    # for i in range(result.shape[0]):
    #     for j in range(result.shape[1]):
    #         if np.all(result[i, j] < 150):
    #             result[i, j] = 0
    #         if np.all(result[i, j] > 210):
    #             result[i, j] = 255

    return result


def show(name, img_show):
    img_show_resize = cv2.resize(img_show, (0, 0), fx=img_resize, fy=img_resize)
    cv2.imshow(name, img_show_resize)
    # cv2.waitKey(0)


img_resize = 1
if __name__ == '__main__':
    img = cv2.imread("asset/image/transform.pgm")
    # rotate_img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rotation = 26
    rotate_img = rotation(img, -img_rotation)
    cv2.imwrite("asset/image/rotation.jpg", rotate_img)
    show("img", img)
    show("transform", rotate_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

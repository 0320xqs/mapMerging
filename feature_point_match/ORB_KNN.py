import cv2
import numpy as np

if __name__ == '__main__':

    # 读取输入图像
    img1 = cv2.imread('asset/image/part01-2.jpg')
    img2 = cv2.imread('asset/image/part02-2.jpg')

    # 创建ORB特征检测器和描述符
    orb = cv2.ORB_create()

    # 在图像上检测关键点和计算描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 0), flags=0)
    cv2.drawKeypoints(img2, kp1, img2, color=(0, 255, 0), flags=0)
    # 创建BFMatcher对象，使用k=2进行kNN匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比例测试，保留好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示匹配结果
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
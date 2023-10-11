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

    # 创建BFMatcher对象，使用k=2进行kNN匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比例测试，保留好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 提取匹配点的关键点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法进行匹配点筛选和变换估计
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 应用变换矩阵将第一幅图像上的角点变换到第二幅图像上
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 绘制匹配结果和RANSAC估计的变换框
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.polylines(img_matches, [np.int32(dst)], True, (0, 255, 0), 3)

    # 显示匹配结果和RANSAC估计的变换框
    cv2.imshow("Matches with RANSAC", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
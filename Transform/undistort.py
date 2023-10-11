import cv2
import numpy as np

if __name__ == '__main__':

    # 读取含有失真的图像
    img = cv2.imread("asset/image/rotation.jpg")
    
    # 读取相机内参矩阵和相机畸变系数
    cameraMatrix = np.array([[2520., 0., 1260.], [0., 2520., 960.], [0., 0., 1.]])
    distCoeffs = np.array([[-0.4, 0.2, 0., 0., 0.]])
    # 进行矫正
    dst = cv2.undistort(img, cameraMatrix, distCoeffs)
    # 显示原图和矫正后的图像
    cv2.imshow("Original Image", img)
    cv2.imshow("Undistorted Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
#读取图像
if __name__ == '__main__':

    img = cv2.imread('asset/image/part01-2.jpg')
    #选择感兴趣区域
    roi = img[0:img.shape[0], 0:img.shape[1]]
    #显示图像和ROI
    cv2.imshow("Image", img)
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
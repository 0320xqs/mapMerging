import cv2
import numpy as np

if __name__ == '__main__':

    img = cv2.imread('right-angle-detect.jpg')  # 这是一张没有直线图案的图片
    # img = cv2.imread('true.jpeg') #这是一张有直线图案的图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 12, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[2]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(lines)
    cv2.imshow("edges", edges)
    cv2.imshow("lines", img)
    cv2.waitKey()
    cv2.destoryAllWindows()

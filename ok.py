import cv2
import matplotlib.pyplot as plt
import numpy as np


def threshold(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thr


def filtering(img_gray, esp):
    if esp == "median":
        return cv2.medianBlur(img_gray, 5)
    elif esp == "gaussian":
        return cv2.GaussianBlur(img_gray, (5, 5), 0)
    elif esp == "bilateral":
        return cv2.bilateralFilter(img_gray, 5, 50, 100)
    else:
        return img_gray


def main():
    img = cv2.imread("./IMG_5327.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = filtering(img, "gaussian")
    #img = filtering(img, "bilateral")
    #img_thresh = threshold(img_filter)

    canny = cv2.Canny(img, 100, 200)

    plt.imshow(canny)
    plt.show()


if __name__  == '__main__':
    main()

import cv2
import matplotlib.pyplot as plt
import numpy as np


def threshold(img_gray, option):
    blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
    if option == "adaptive":
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
    if option == "global":
        _, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        return img
    else:
        return img_gray


def filtering(img_gray, esp):
    if esp == "median":
        return cv2.medianBlur(img_gray, 5)
    elif esp == "gaussian":
        return cv2.GaussianBlur(img_gray, (5, 5), 0)
    elif esp == "bilateral":
        return cv2.bilateralFilter(img_gray, 5, 50, 100)
    else:
        return img_gray


def Gamma(img_gray, gamma):
    return (np.power(img_gray / 255, gamma).clip(0, 1) * 255).astype(np.uint8)

scale = 1.0
img = cv2.imread("./jpg/Templates/1zl.jpg")
img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
width, height = img.shape[:2]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = Gamma(img, 0.20)
img = threshold(img, "adaptive")
img = cv2.Canny(img, 100, 200)

plt.imshow(img)
plt.show()
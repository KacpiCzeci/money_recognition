import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    return np.power(img_gray, gamma).clip(0, 255)

image1 = cv2.imread('./IMG_5340.jpg')
image2 = cv2.imread('./50zl1.jpg')


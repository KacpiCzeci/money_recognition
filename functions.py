import cv2
import matplotlib.pyplot as plt
import numpy as np

scale = 6.0

def crop(image, rect):
   shape = (image.shape[1], image.shape[0])
   w, h = rect[1]
   center = (rect[0][0], rect[0][1])
   M = cv2.getRotationMatrix2D(center, rect[2], 1.0)
   rotated_image = cv2.warpAffine(image, M, shape)
   x = int(center[0] - w/2)
   y = int(center[1] - h/2)
   return rotated_image[y:y+int(h), x:x+int(w)]


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

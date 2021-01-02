from pylab import *
import cv2
class Filter():

    def __init__(self, img):
        self.image = img
        self.original = img

    def crop(self, rect):
        shape = (self.image.shape[1], self.image.shape[0])
        w, h = rect[1]
        center = (rect[0][0], rect[0][1])
        M = cv2.getRotationMatrix2D(center, rect[2], 1.0)
        rotated_image = cv2.warpAffine(self.image, M, shape)
        x = int(center[0] - w / 2)
        y = int(center[1] - h / 2)
        return rotated_image[y:y + int(h), x:x + int(w)]

    def resize(self, scale):
        self.image = cv2.resize(self.image, (int(self.image.shape[1]/scale), int(self.image.shape[0]/scale)))

    def mor

    def threshold(self, option):
        blur = cv2.GaussianBlur(self.image, (15, 15), 0)
        if option == "adaptive":
            self.image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        if option == "global":
            _, img = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)
            self.image = img

    def filtering(self, esp):
        if esp == "median":
            self.image = cv2.medianBlur(self.image, 5)
        elif esp == "gaussian":
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        elif esp == "bilateral":
            self.image = cv2.bilateralFilter(self.image, 5, 50, 100)

    def Gamma(self, gamma):
        self.image = (np.power(self.image / 255, gamma).clip(0, 1) * 255).astype(np.uint8)
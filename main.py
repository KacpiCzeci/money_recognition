import cv2
import matplotlib.pyplot as plt
import numpy as np
from loadfiles import fileManager
from inputFilters import Filter

scale = 8.0

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


def main():
    img = cv2.imread("./jpg/Medium/IMG_5399.jpg")
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = Gamma(img, 0.5)
    img = threshold(img, "adaptive")
    plt.imshow(img)
    plt.show()
    kernel = np.ones((3, 3), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=3)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    plt.imshow(img)
    plt.show()
    img = cv2.Canny(img, 100, 200)
    plt.imshow(img)
    plt.show()

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_filtered = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 25*25 and w*h < 150*150 :
            temp_img = cv2.drawContours(temp_img, contour, -1, (255, 0, 0), 5)
            #temp_img = cv2.fillPoly(temp_img, contours, (255, 255, 255))
            temp = cv2.fitEllipse(contour)
            temp = ((temp[0][0]*scale, temp[0][1]*scale), (temp[1][0]*scale, temp[1][1]*scale), temp[2])
            original = cv2.ellipse(original, temp, (255, 0, 0), 30)
            #plt.imshow(temp_img[y:y + h, x:x + w])
            #plt.show()

    plt.imshow(temp_img)
    plt.show()
    plt.imshow(original)
    plt.show()

if __name__  == '__main__':
    ##wstep
    main()
    IOtool = fileManager()
    IOtool.loadFile()
    imSet = IOtool.getImageSet()
    FilterTool = Filter(imSet)
    ###rozwiniecie

    FilterTool.blackAndWhite()



    ###zakonczenie
    IOtool.setImageSet(FilterTool.getImageSet())
    IOtool.saveFile()
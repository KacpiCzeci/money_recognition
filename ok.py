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


def main():
    '''
    coins = {
        "1gr": 1.000000000000,
        "10gr": 1.064516129032,
        "1zl": 1.483870967742,
        "2zl": 1.387096774194,
        "5zl": 1.548387096774
    }
    '''
    coins = {
        "1gr": 15.5,
        "10gr": 16.5,
        "1zl": 23.0,
        "2zl": 21.5,
        "5zl": 24.0
    }
    #'''
    img = cv2.imread("./jpg/Easy/IMG_5401.jpg")
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    width, height = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = Gamma(img, 0.50)
    img = threshold(img, "adaptive")
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=4)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    plt.imshow(img)
    plt.show()
    img = cv2.Canny(img, 100, 200)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        img = cv2.drawContours(img, [contour], 0, 255, -1)

    plt.imshow(img)
    plt.show()

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fit_ellipses = []
    crop_ellipses = []
    match_to_coin = []
    match_size = []

    for contour in contours:
        min = cv2.minAreaRect(contour)
        if min[1][0]*min[1][1] > width*height/100 and min[1][0]*min[1][1] < width*height/10:
            temp = cv2.fitEllipse(contour)
            if temp[1][0] / temp[1][1] < 2.5 and temp[1][1] / temp[1][0] < 2.5:
                min = ((min[0][0] * scale, min[0][1] * scale), (min[1][0] * scale, min[1][1] * scale), min[2])
                coin_check = crop(cv2.cvtColor(original, cv2.COLOR_RGB2HSV), min)
                if abs(cv2.mean(coin_check)[0] - 25.0) <= 10.0:
                    temp = ((temp[0][0]*scale, temp[0][1]*scale), (temp[1][0]*scale, temp[1][1]*scale), temp[2])
                    fit_ellipses.append(temp)
                    match_to_coin.append(False)
                    match_size.append(0.0)
                    crop_ellipses.append(cv2.cvtColor(coin_check, cv2.COLOR_HSV2RGB))
                    print(cv2.mean(coin_check))

    sorted(fit_ellipses, key=lambda ell: ell[1][0] * ell[1][1] * np.pi)

    starting_size = width*height/100
    ending_size = width*height/10


    for key, value in coins.items():
        for i in range(len(fit_ellipses)):
            ellipse = fit_ellipses[i]
            if key == "1gr":
                match_to_coin[i] = "1gr"
                temp = max(ellipse[1][0], ellipse[1][1])
                match_size[i] = abs(int(temp/value)-temp/value)
            else:
                temp = max(ellipse[1][0], ellipse[1][1])
                temp = abs(int(temp/value) - temp/value)
                if temp < match_size[i]:
                    match_to_coin[i] = key
                    match_size[i] = temp
    '''

    while starting_size < ending_size:
    for key, value in coins.items():
        for i in range(len(fit_ellipses)):
            ellipse = fit_ellipses[i]
            if key == "1gr":#
                match_to_coin[i] = "1gr"#
                temp = max(ellipse[1][0], ellipse[1][1])#
                match_size[i] = abs(int(temp/value)-temp/value)

            d1 = abs(starting_size * value - ellipse[1][0])#
            d2 = abs(starting_size * value - ellipse[1][1])#
            if d1 < value or d2 < value:
                if (match_to_coin[i]) == False:
                    match_to_coin[i] = key
                    match_size[i] = starting_size
                    #starting_size = ending_size
                else:
                    previous_value = coins.get(match_to_coin[i])
                    d11 = abs(match_size[i] * previous_value - ellipse[1][0])
                    d22 = abs(match_size[i] * previous_value - ellipse[1][1])
                    if d11 > d1 or d22 > d2:
                    #if previous_value < value:
                        match_to_coin[i] = key
                        match_size[i] = starting_size

        starting_size += 0.01
        
        
    for ellipse in crop_ellipses:
        plt.imshow(ellipse)
        plt.show()
    #'''

    template = cv2.imread("jpg/Templates/10gr.jpg")
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #template = cv2.resize(template, (int(template.shape[1] / 2.0), int(template.shape[0] / 2.0)))
    #template = cv2.GaussianBlur(template, (5, 5), 5)
    for i in range(2):
        template = cv2.pyrDown(template)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
    for ellipse in crop_ellipses:
        ellipse = cv2.cvtColor(ellipse, cv2.COLOR_BGR2GRAY)
        for i in range(1):
            ellipse = cv2.pyrDown(ellipse)
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(ellipse, None)
        matches = bf.match(des1, des2)
        img3 = cv2.drawMatches(template, kp1, ellipse, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        print(len(kp1), len(kp2), len(matches))
        plt.imshow(img3)
        plt.show()

    for i in range(len(fit_ellipses)):
        if match_to_coin[i]:
            original = cv2.ellipse(original, fit_ellipses[i], (255, 0, 0), 30)
            x, y = fit_ellipses[i][0]
            original = cv2.putText(original, match_to_coin[i], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10, cv2.LINE_AA)

    plt.imshow(original)
    plt.show()


if __name__  == '__main__':
    main()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import functions as f

def main():
    '''
    coins = {
        "1gr": 1.000000000000,
        "10gr": 1.064516129032,
        "1zl": 1.483870967742,
        "2zl": 1.387096774194,
        "5zl": 1.548387096774
    }

    coins = {
        "1gr": 15.5,
        "10gr": 16.5,
        "1zl": 23.0,
        "2zl": 21.5,
        "5zl": 24.0
    }
    '''
    img = cv2.imread("./jpg/Easy/IMG_5418.jpg")
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(img.shape[1]/f.scale), int(img.shape[0]/f.scale)))
    width, height = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = f.Gamma(img, 0.50)
    img = f.threshold(img, "adaptive")
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=4)
    img = cv2.Canny(img, 100, 200)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        img = cv2.drawContours(img, [contour], 0, 255, -1)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fit_ellipses = []
    crop_ellipses = []
    match_to_coin = []
    match_size = []
    match_ratio = []

    for contour in contours:
        min = cv2.minAreaRect(contour)
        if min[1][0]*min[1][1] > width*height/100 and min[1][0]*min[1][1] < width*height/10:
            temp = cv2.fitEllipse(contour)
            if temp[1][0] / temp[1][1] < 2.5 and temp[1][1] / temp[1][0] < 2.5:
                min = ((min[0][0] * f.scale, min[0][1] * f.scale), (min[1][0] * f.scale, min[1][1] * f.scale), min[2])
                coin_check = f.crop(cv2.cvtColor(original, cv2.COLOR_RGB2HSV), min)
                if abs(cv2.mean(coin_check)[0] - 25.0) <= 10.0:
                    temp = ((temp[0][0]*f.scale, temp[0][1]*f.scale), (temp[1][0]*f.scale, temp[1][1]*f.scale), temp[2])
                    fit_ellipses.append(temp)
                    match_to_coin.append(False)
                    match_size.append(0.0)
                    crop_ellipses.append(cv2.cvtColor(coin_check, cv2.COLOR_HSV2RGB))
                    match_ratio.append(0)

    #sorted(fit_ellipses, key=lambda ell: ell[1][0] * ell[1][1] * np.pi)

    #starting_size = width*height/100
    #ending_size = width*height/10

    templates = ["1gr", "10gr", "1zl", "2zl", "5zl"]
    for name in templates:
        template = cv2.imread("jpg/Templates/{}.jpg".format(name))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        for i in range(2):
            template = cv2.pyrDown(template)
        orb = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        for n, ellipse in enumerate(crop_ellipses):
            ellipse = cv2.cvtColor(ellipse, cv2.COLOR_BGR2GRAY)
            for i in range(0):
                ellipse = cv2.pyrDown(ellipse)
            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(ellipse, None)
            if kp1 == [] or kp2 == []:
                continue
            matches = bf.match(des1, des2)
            img3 = cv2.drawMatches(template, kp1, ellipse, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            ratio = len(matches) #/ len(kp1)
            if match_ratio[n] < ratio:
                match_ratio[n] = ratio
                match_to_coin[n] = name
            print(len(kp1), len(kp2), len(matches))

    for i in range(len(fit_ellipses)):
        if match_to_coin[i]:
            original = cv2.ellipse(original, fit_ellipses[i], (255, 0, 0), 30)
            x, y = fit_ellipses[i][0]
            original = cv2.putText(original, match_to_coin[i], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10, cv2.LINE_AA)

    print(match_ratio)
    plt.imshow(original)
    plt.show()


if __name__  == '__main__':
    main()
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import functions as f
import constants as c
import sys


def main():
    img = cv2.imread("./jpg/{}".format(sys.argv[1]))
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(img.shape[1]/f.scale), int(img.shape[0]/f.scale)))
    width, height = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.addWeighted(img, 1.2, img, 0, 1)

    img = f.Gamma(img, 0.50)
    img = f.threshold(img, "adaptive")
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.Canny(img, 100, 200)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (255, 255, 255), cv2.FILLED)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fit_ellipses = []
    crop_ellipses = []
    match_to_coin = []

    ratios = []
    for contour in contours:
        minA = cv2.minAreaRect(contour)
        if minA[1][0] * minA[1][1] > (width * height) / 150:
            temp = cv2.fitEllipse(contour)
            minA = ((minA[0][0] * f.scale, minA[0][1] * f.scale), (minA[1][0] * f.scale, minA[1][1] * f.scale), minA[2])
            coin_check = f.crop(original, minA)
            coin_check = cv2.cvtColor(coin_check, cv2.COLOR_RGB2HSV)
            temp = ((temp[0][0] * f.scale, temp[0][1] * f.scale), (temp[1][0] * f.scale, temp[1][1] * f.scale), temp[2])
            ratios.append(max(temp[1][0]/temp[1][1], temp[1][1]/temp[1][0]))
            fit_ellipses.append(temp)
            match_to_coin.append(False)
            crop_ellipses.append(coin_check)

    reference = min(ratios, key=lambda rt: abs(1-rt))
    new_fit_ellipses = []
    new_crop_ellipses = []
    new_match_to_coin = []
    for i in range(len(fit_ellipses)):
        if abs(ratios[i]-reference) < 0.40:
            new_fit_ellipses.append(fit_ellipses[i])
            new_crop_ellipses.append(crop_ellipses[i])

            new_match_to_coin.append(match_to_coin[i])

    fit_ellipses = new_fit_ellipses
    crop_ellipses = new_crop_ellipses
    match_to_coin = new_match_to_coin

    silver_in = silver_out = bronze_in = bronze_out = 0

    for i, ellipse in enumerate(crop_ellipses):
        ellipse = cv2.cvtColor(ellipse, cv2.COLOR_RGB2GRAY)
        m = np.mean(ellipse)
        ellipse = cv2.addWeighted(ellipse, c.contrast, ellipse, 0, (c.mean_tresh - round(m, 0)))
        h = int(ellipse.shape[0]/2.0)
        w = int(ellipse.shape[1]/2.0)
        plt.imshow(ellipse)
        plt.show()
        for y in range(len(ellipse)):
            for x in range(len(ellipse[0])):
                r = (x-w)*(x-w)/(w*w)+(y-h)*(y-h)/(h*h)
                if r <= 1 and r > 0.4:
                    pixel = ellipse[y][x]
                    if pixel >= c.step:
                        ellipse[y][x] = 255
                        bronze_out += 1
                    else:
                        ellipse[y][x] = 0
                        silver_out += 1
                elif r <= 0.4:
                    pixel = ellipse[y][x]
                    if pixel >= c.step:
                        ellipse[y][x] = 255
                        bronze_in += 1
                    else:
                        ellipse[y][x] = 0
                        silver_in += 1

        if silver_in > bronze_in and silver_out > bronze_out:
            match_to_coin[i] = "10gr_or_1zl"
        elif silver_in < bronze_in and silver_out < bronze_out:
            match_to_coin[i] = "1gr"
        elif silver_in > bronze_in and silver_out < bronze_out:
            match_to_coin[i] = "2zl"
        elif silver_in < bronze_in and silver_out > bronze_out:
            match_to_coin[i] = "5zl"
        silver_in = silver_out = bronze_in = bronze_out = 0

    occurences = []
    checked = False
    to_check = [i for i, x in enumerate(match_to_coin) if x == "10gr_or_1zl"]
    if to_check:
        for value in ["1gr", "2zl", "5zl"]:
            occurences = [i for i, x in enumerate(match_to_coin) if x == value]
            if not occurences:
                continue
            else:
                size_min = np.mean([min(fit_ellipses[i][1][0], fit_ellipses[i][1][1]) for i in occurences])
                size_max = np.mean([max(fit_ellipses[i][1][0], fit_ellipses[i][1][1]) for i in occurences])
                for check in to_check:
                    check_size_min = min(fit_ellipses[check][1][0], fit_ellipses[check][1][1])
                    check_size_max = max(fit_ellipses[check][1][0], fit_ellipses[check][1][1])
                    if check_size_min > 1.6 * size_min or check_size_min < 0.6 * size_min or check_size_max > 1.6 * size_max or check_size_max < 0.6 * size_max:
                        match_to_coin[check] = False
                        continue
                    elif value == "5zl" and (check_size_min > 1.3 * size_min or check_size_max > 1.3 * size_max):
                        match_to_coin[check] = False
                        continue
                    ratio = size_max / check_size_max
                    if abs(c.coin_sizes.get(value)/c.coin_sizes.get("10gr")-ratio) < abs(c.coin_sizes.get(value)/c.coin_sizes.get("1zl")-ratio):
                        match_to_coin[check] = "10gr"
                    else:
                        match_to_coin[check] = "1zl"
                checked = True

    if not checked:
        for check in to_check:
            size = max(fit_ellipses[check][1][0], fit_ellipses[check][1][1])
            ratio10 = size/c.coins_values.get("10gr")
            ratio1 = size/c.coins_values.get("1zl")
            if abs(int(ratio10)-ratio10) < abs(int(ratio1)-ratio1):
                match_to_coin[check] = "10gr"
            else:
                match_to_coin[check] = "1zl"

    for i in range(len(fit_ellipses)):
        if match_to_coin[i]:
            original = cv2.ellipse(original, fit_ellipses[i], (255, 0, 0), 30)
            x, y = fit_ellipses[i][0]
            _, h = fit_ellipses[i][1]
            original = cv2.putText(original, match_to_coin[i], (int(x), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10, cv2.LINE_AA)
    sum = 0
    for coin in match_to_coin:
        if coin:
            sum += c.coins_values.get(coin)
    original = cv2.putText(original, "Suma: {} zl".format(round(sum, 2)), (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10, cv2.LINE_AA)

    plt.imshow(original)
    plt.show()

    cv2.imwrite("./results/{}".format(sys.argv[1]), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()

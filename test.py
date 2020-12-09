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


image1 = cv2.imread('./IMG_5340.jpg')
image2 = cv2.imread('./50zl1.jpg')

test = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#test = cv2.GaussianBlur(image1, (0, 0), sigmaX=np.sqrt(2.0)*0.5, sigmaY=np.sqrt(2.0)*0.5)
test = filtering(test, "bilateral")
test = cv2.Canny(test, 100, 200)

test = cv2.pyrDown(test)
test = cv2.pyrDown(test)
#test = cv2.pyrDown(test)
#test = cv2.pyrDown(test)

#test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)

template = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
template = filtering(template, "bilateral")
template = cv2.Canny(template, 100, 200)
template = cv2.pyrDown(template)
template = cv2.pyrDown(template)
template = cv2.pyrDown(template)
#template = cv2.pyrDown(template)
num_rows, num_cols = template.shape[:2]

#template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

test_keypoints, test_descriptor = sift.detectAndCompute(test, None)
template_keypoints, template_descriptor = sift.detectAndCompute(template, None)

keypoints_without_size = np.copy(test)
keypoints_with_size = np.copy(test)

cv2.drawKeypoints(test, test_keypoints, keypoints_without_size, color = (0, 255, 0))

cv2.drawKeypoints(test, test_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].imshow(keypoints_with_size, cmap='gray')
plots[1].imshow(keypoints_without_size, cmap='gray')


bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)

matches = bf.match(test_descriptor, template_descriptor)

matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(test, test_keypoints, template, template_keypoints, matches, template, flags = 2)

plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.imshow(result)
plt.show()

print("Test keypoints: ", len(test_keypoints))
print("Template keypoints: ", len(template_keypoints))
print("Matched keypoints: ", len(matches))
if len(matches) >= 0.75 * len(template_keypoints):
    print("Match")
else:
    print("Not match")
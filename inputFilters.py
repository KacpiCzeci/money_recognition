import numpy as np
import warnings
from pylab import *
import skimage as ski
from skimage import img_as_float
from skimage.color import rgb2gray
class Filter():

    def __init__(self,image):
        self.imageSet = image


#Funkcja do oblicznia wartosci progowej
    def thresh(self,t):
        ite = 0
        new_set = []
        for image in self.imageSet:
            warnings.simplefilter("ignore")
            binary = (image > t) * 255
            binary = np.uint8(binary)
            new_set.append(binary)
            ite += 1
        self.imageSet = new_set

    #nietestowane
    def setGamma(self,gamma):
        new_set = []
        for image in self.imageSet:
            image = img_as_float(image)
            new_set.append(image ** gamma)
        self.imageSet = new_set

    def blackAndWhite(self):
        new_set = []
        for img in self.imageSet:
            img = rgb2gray(img)
            img = img_as_float(img)
            new_set.append(img)
        self.imageSet = new_set


        warnings.simplefilter("ignore")
    def getImageSet(self):
        return self.imageSet
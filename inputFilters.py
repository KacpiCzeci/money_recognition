import numpy as np
import warnings
import copy
from pylab import *
import skimage as ski
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage import util
import skimage.morphology as mp
from skimage import filters
import numpy as np


class Filter():

    def __init__(self,image):
        self.imageSet = image
        self.spanOfObjectRecognition = 10
        self.objectsAtImages = []
        warnings.simplefilter("ignore")


#Funkcja do oblicznia wartosci progowej
    def thresh(self,t):
        ite = 0
        new_set = []
        for image in self.imageSet:
            warnings.simplefilter("ignore")
            binary = (image > mean(image)+t) * 255
            binary = np.uint8(binary)
            new_set.append(binary)
            ite += 1
        self.imageSet = new_set

    #nietestowane
    def setGamma(self,gamma):
        new_set = []
        for image in self.imageSet:
            new_set.append(image ** gamma)
        self.imageSet = new_set

    def blackAndWhite(self):
        new_set = []
        for img in self.imageSet:
            img = rgb2gray(img)
            img = img_as_float(img)
            new_set.append(img)
        self.imageSet = new_set

    # Funkcja do oblicznia kontrastu, wartosci MIM i MAX to 2.3 odchylenia standardowego od sredniej
    def Contrast(self):
        new_set = []
        for img in self.imageSet:
            mean = np.mean(img)
            std = np.std(img)
            MIN = mean - 2.3 * std
            MAX = mean + 2.3 * std
            if MIN < np.percentile(img, 0):
                MIN = np.percentile(img, 5)
            if MAX > np.percentile(img, 100):
                MAX = np.percentile(img, 95)
            norm = (img - MIN) / (MAX - MIN)
            norm[norm[:, :] > 1] = 1
            norm[norm[:, :] < 0] = 0
            new_set.append(norm)
        warnings.simplefilter("ignore")
        self.imageSet = new_set

        #Negatyw i dopelnienie obiektow za pomoca dylacji
    def negative(self):
        new_set = []
        for img in self.imageSet:
            img = util.invert(img)
            new_set.append(img)
        self.imageSet = new_set

    def dilation(self):
        new_set = []
        for img in self.imageSet:
            #img = mp.dilation(img, selem=K)
            img = mp.dilation(img)
            new_set.append(img)
        self.imageSet = new_set

    def sobel(self):
        new_set = []
        for img in self.imageSet:
            img = filters.sobel(img)
            new_set.append(img)
        self.imageSet = new_set


    #algorytm znajduje klastry białych obiektów i wycina je z obrazka (obrazek musi zawierać tylko
    # wartości 255 i 0, oraz być dwuwymiarowy)
    def object_finder(self):

        for img in self.imageSet:
            shape = img.shape
            imgCP = copy(img)
            for col in range(shape[0]):
                for row in range(shape[1]):
                    if imgCP[col][row] == 255:
                        impArea = self.Recognise(col,row,imgCP,shape[1],shape[0])
                        up = impArea[0]
                        down = impArea[1]
                        right = impArea[2]
                        left = impArea[3]
                        #ten if odrzuca znalezione klastry które są za małe
                        if(right-left)*(down-up) >= 5000:
                            #kopiujemy części oryginało, a na kopi je wymazujemy żeby nie przetwarzać
                            #ich ponownie w następnych iteracjach
                            self.objectsAtImages.append(img[up:down, left:right])
                            for pixelX in range(left, right):
                                for pixelY in range(up, down):
                                    imgCP[pixelY][pixelX] = 100

    #próbujemy dany punkt przemienić w kwadrat, który pokrywa jak najwięcej białych elementów
    #na obrazku
    def Recognise(self,col,row, img, maxRight, maxDown):
        left = row - self.spanOfObjectRecognition
        if left < 0:
            left = 0

        right = row + self.spanOfObjectRecognition
        if right > maxRight-1:
            right = maxRight-1

        up = col - self.spanOfObjectRecognition
        if up < 0:
            up = 0

        down = col + self.spanOfObjectRecognition
        if down > maxDown-1:
            down = maxDown-1


        #pętla while na początku jest wyłączana, a wykonanie któregokolwiek z ifów sprawie, że
        #zrobi następny obrót
        changed = 1
        changeOfSearch = 10

        while changed > 0:
            changed = 0
            for pixel in range(left, right):
                if img[up][pixel] == 255 and up-changeOfSearch >= 0:
                    up -= changeOfSearch
                    changed += 1
                    break
                elif img[down][pixel] == 255 and down+changeOfSearch <= maxDown:
                    down += changeOfSearch
                    changed += 1
                    break
            for pixel in range(up, down):
                if img[pixel][left] == 255 and left-changeOfSearch >= 0:
                    left -= changeOfSearch
                    changed += 1
                    break
                elif img[pixel][right] == 255 and right+changeOfSearch <= maxRight:
                    right += changeOfSearch
                    changed += 1
                    break
        return((up,down,right,left))




    def getObjectsAtImg(self):
        return self.objectsAtImages

    def getImageSet(self):
        return self.imageSet

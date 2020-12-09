from skimage.io import imread_collection
from skimage.io import imsave
import os
import imageio


class fileManager:

    def __init__(self):
        self.imageSet = []
        self.inputPath = ".\\input\\*.jpg"
        self.outputPath = ".\\output"

    def loadFile(self):
        self.imageSet = imread_collection(self.inputPath)

    def saveFile(self):
        #try:
            #os.rmdir(self.outputPath)
        #except IOError:
            #print("try again after creating output dir in your project")
        orderNumber = 0
        for image in self.imageSet:
            orderNumber += 1
            imName = self.outputPath + f"\\output {orderNumber}.jpg"
            imsave(imName, image)

    def setImageSet(self,img):
        self.imageSet = img

    def getImageSet(self):
        return self.imageSet
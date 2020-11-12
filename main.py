from pylab import *
import skimage as ski
from skimage import data, io, filters
from skimage.filters import rank
from skimage import img_as_float
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2gray
from matplotlib import pylab as plt
import numpy as np
from skimage import feature
from skimage import measure
from numpy import array
import warnings
from pathlib import Path
from loadfiles import fileManager
from inputFilters import Filter

#Ladowanie pliku ze sciezkami do obrazow
def loadfile():
    try:
        file = open("filenames.txt", "r")
    except IOError:
        print("Could not open file!\n")
        return -1

    data = list()
    for line in file:
        path = Path(line.rstrip("\n"))
        if path.is_file():
            data.append(line.rstrip("\n"))

    file.close()
    return data

#Ladowanie obrazu
def loadimage (name):
    return ski.io.imread(name)





#Funkcja do oblicznia kontrastu, wartosci MIM i MAX to 2.3 odchylenia standardowego od sredniej
def contrast(img):
    mean  = np.mean(img)
    std = np.std(img)
    MIN = mean - 2.3*std
    MAX = mean + 2.3*std
    if MIN < np.percentile(img, 0):
        MIN = np.percentile(img, 5)
    if MAX > np.percentile(img, 100):
        MAX = np.percentile(img, 95)
    norm = (img - MIN) / (MAX - MIN)
    norm[norm[:,:] > 1] = 1
    norm[norm[:,:] < 0] = 0
    return norm


def main():
    filenames = loadfile()
    if(filenames == -1):
        return 1

    planes = list()

    for file in filenames:
        planes.append(loadimage(file))

    #Utworzenie dwoch oddzielnych prestrzeni (na zad na 3 i na 5)
    size = (2, 3)
    fig1, ax1 = plt.subplots(size[0], size[1], figsize=(48, 24))
    fig2, ax2 = plt.subplots(size[0], size[1], figsize=(48, 24))
    plot_index = [0, 0]
    i = 0
    saved = False
    for img in planes:
        temp = img
        K = np.ones((6,6))



        #ustawienie kontrastu i wartosci progowej
        img = contrast(img_as_float(img))
        img = thresh(mean(img)-0.9*mean(img), img)

        #Negatyw i dopelnienie obiektow za pomoca dylacji
        img = util.invert(img)
        img = mp.dilation(img, selem=K)
        #Okreslenie regionow, rozniacych sie barwami
        label_img = label(img)
        regions = regionprops(label_img)
        img = util.invert(img)

        #Wygladzenie obrazu
        img = filters.median(img, disk(5))
        img = filters.gaussian(img, sigma=2)

        #Filtr wykrywajacy kolory
        img1 = filters.sobel(img)
        img1 = setGamma(5.0, img1)
        ax1[plot_index[0], plot_index[1]].imshow(img1, cmap=plt.cm.gray, interpolation='nearest')

        #Wykrycie i dodanie konturow (kolor)
        img2 = setGamma(5.0, img)
        contours = measure.find_contours(img2, 0.8)
        for contour in contours:
            ax2[plot_index[0], plot_index[1]].plot(contour[:, 1], contour[:, 0], linewidth=2)

        # Okreslenie i dodanie centroidow
        for props in regions:
            y, x = props.centroid
            ax2[plot_index[0], plot_index[1]].plot(x, y, '.w', markersize=5)
        ax2[plot_index[0], plot_index[1]].imshow(temp, interpolation='nearest')


        #zapisz do plików po 6 obrazów, BW - zad na 3, Cl - zad na 5
        if plot_index[0] == size[0]-1 and plot_index[1] == size[1] - 1:
            fig1.savefig('BW{}.pdf'.format(i))
            fig2.savefig('CL{}.pdf'.format(i))
            i += 1
            plot_index[0] = 0
            plot_index[1] = 0
            saved = True
            for n in range(size[0]):
                for m in range(size[1]):
                    ax1[n, m].clear()
                    ax2[n, m].clear()
        elif (plot_index[1]+1)%size[1] == 0:
            plot_index[0] += 1
            plot_index[1] = 0
            saved = False
        else:
            plot_index[1] += 1
            saved = False
    if not saved:
        fig1.savefig('BW{}.pdf'.format(i+1))
        fig2.savefig('CL{}.pdf'.format(i+1))


if __name__  == '__main__':
    ##wstep
    IOtool = fileManager()
    IOtool.loadFile()
    imSet = IOtool.getImageSet()
    FilterTool = Filter(imSet)
    ###rozwiniecie

    FilterTool.blackAndWhite()



    ###zakonczenie
    IOtool.setImageSet(FilterTool.getImageSet())
    IOtool.saveFile()
import numpy as np
from PIL import Image
import csv
import colored

def loadImages(dir,amount = 10,size = 1,verbal = False):
    images = []
    for i in range(amount):
        img = Image.open(dir + "img" + str(i) + ".png") #load the image
        img.thumbnail((img.size[0]*size,img.size[1]*size),Image.ANTIALIAS) #resize the image
        images.append(np.array(img)) #add to array
    

    if(verbal):
        if(len(images) > 0):
            print("\nFound",len(images),"images.\n")
        else:
            print("\n//ERR// Load Error. Make sure your images are named 'img0000', where 0000 is the number of image corresponding to (label) csv row\n")
    return np.array(images)

def divideData(data,ratio = 0.5):
    return data[:int(len(data) * ratio)], data[int(len(data) * ratio):]

def getLabels(amount = 10):
    with open('../data/data.csv',newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        labels = []

        for row in reader:
            labels.append(row[1:])
        
        if(len(labels) < amount):
            return np.array(labels)
        return np.array(labels[:amount])

messageColor = colored.fg(118)
errorColor = colored.fg(196)
noticeColor = colored.fg(4)
spamColor = colored.fg(245)
resetColor = colored.attr('reset')
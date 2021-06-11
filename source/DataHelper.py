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

def waitForExit():
    print(messageColor)
    input("\nPress Enter to exit...")
    print(resetColor)

def getLabels(amount = 10):
    with open('../data/data.csv',newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        labels = []

        for row in reader:
            vals = []
            for val in row[1:]:
                if(str(val).isnumeric()):
                    vals.append(float(val))
            labels.append(vals)
        
        if(len(labels) < amount):
            return np.array(labels)
        return np.array(labels[:amount])

def getData(amount = 10,imgSize = 1,yPosition = 1):
    with open('../data/data.csv','r',newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        yData = []

        xData = []

        
        valTypes = None
        
        first = True

        for row in reader:
            if(first):
                first = False
                valTypes = row
                continue

            vals = []
            
            for i in range(len(row[yPosition:])):
                val = row[i+yPosition]
                valType = valTypes[i+yPosition]
                if(valType == 'int'):
                    vals.append(int(val))
                elif(valType == 'float'):
                    vals.append(float(val))
                elif(valType == 'image'):
                    img = Image.open('../data/'+val) #load the image
                    img.thumbnail((img.size[0]*imgSize,img.size[1]*imgSize),Image.ANTIALIAS) #resize the image
                    vals.append(np.array(img)) #add to array

            yData.append(vals)

            vals = []
            for i in range(len(row[:yPosition])):
                val = row[i]
                valType = valTypes[i]
                if(valType == 'int'):
                    vals.append(int(val))
                elif(valType == 'float'):
                    vals.append(float(val))
                elif(valType == 'image'):
                    img = Image.open('../data/'+val) #load the image
                    img.thumbnail((img.size[0]*imgSize,img.size[1]*imgSize),Image.ANTIALIAS) #resize the image
                    vals.append(np.array(img)) #add to array

            xData.append(vals)

        
        xData = np.asarray(xData)
        yData = np.asarray(yData)

        print(xData.shape,yData.shape)

        if(len(yData) < amount):
            return xData ,yData
        return xData[:amount] ,yData[:amount]
        
messageColor = colored.fg(118)
errorColor = colored.fg(196)
noticeColor = colored.fg(4)
spamColor = colored.fg(245)
resetColor = colored.attr('reset')
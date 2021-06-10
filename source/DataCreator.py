import random
import cv2
import csv
import numpy as np

datalen = 3000

back = np.zeros((500,500),np.uint8)
with open('data.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|')

    for i in range(datalen):
        Xpos = random.uniform(0,1)
        Ypos = random.uniform(0,1)

        imname = 'IMGS/img' + str(i) + '.png'


        randBit = random.getrandbits(1)
        if(not bool(randBit)):
            cv2.circle(back,(int(Xpos * 400) + 50, int(Ypos * 450) + 50),10,(255,255,255),-1)
        else:
            cv2.rectangle(back,((int(Xpos * 400) + 50, int(Xpos * 450) + 50)),((int(Xpos * 400) + 70, int(Xpos * 450) + 70)),(255,255,255),-1)
        cv2.imwrite(imname,back)

        back = np.zeros((500,500),np.uint8)

        # writer.writerow([imname + ',',(int(Xpos * 400) + 50) / 500])
        writer.writerow([imname + ',',randBit])

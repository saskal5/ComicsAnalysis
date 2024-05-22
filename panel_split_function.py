import cv2
import numpy as np
import pandas as pd

book = 'T_23'
start = 1
end = 62

image_number = 0

for i in range(start,end+1):
    if (i < 10):
        i_str = str(i)
        pic_name = 'PDFs_HQ/' + book + '/Pages/0' + i_str
    else:
        i_str = str(i)
        pic_name = 'PDFs_HQ/' + book + '/Pages/' + i_str
        
    image = cv2.imread(pic_name + '.jpg')
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    sorted_cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1] * 10 )
    
    for c in sorted_cnts:
        x,y,w,h = cv2.boundingRect(c) 

        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1
        print(x,y)


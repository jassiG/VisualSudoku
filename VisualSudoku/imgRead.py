#from imutils.perspective import four_point_transform
#from imutils import contours
'''
for the loading animation
'''
import sys
import time
import threading

'''
for the rest of the code
'''

import imutils
import cv2
import numpy as np
import pytesseract as pt
import re
#import four_point as fp

desired = {'1','2','3','4','5','6','7','8','9'}
d = []
t1, t2 = time.time(), time.time()
#def animate():
#    i=6
def animated_loading():
    chars = "/—\|" 
    for char in chars:
        sys.stdout.write('\r'+'Breaking into parts and recognizing the numbers...'+char)
        time.sleep(.1)
        sys.stdout.flush() 

def rectify(h):
    '''
    This function takes a numpy array and 
    finds the corners for the actual sudoku puzzle. 
    '''
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


pt.pytesseract.tesseract_cmd = r'C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
#import numpy as np


def get_puzzle(image):
    h, w, ch= image.shape
    #cv2.imshow('before', image)
    blur = cv2.resize(image, None, fx = 449/h, fy = 449/w, interpolation = cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(blur, (1, 1), 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 5, 40, 40)
    thresh = cv2.adaptiveThreshold(grey, 255, 1, 1, 11, 2)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (1, 1))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #thresh = 255 - thresh
    return thresh


def warp_img(image):
    #cv2.imshow('after', image)
    #print(f"h is : {h}  \n w is : {w} ")
    #h, w = image.shape
    #image = cv2.resize(image, None, fx = 449/h, fy = 449/w, interpolation = cv2.INTER_NEAREST)
    conts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)
    for c in conts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            approx = rectify(approx)
            h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
            retval = cv2.getPerspectiveTransform(approx, h)
            warp = cv2.warpPerspective(image, retval, (450, 450))
            warp = 255 - warp
            return warp

def break_parts(image):
    global d
    arr = np.split(image, 9, 1)
    #z=0
    for i in range(9):
        arr2 = np.split(arr[i], 9)
        for j in range(9):
            #for a in range(len(arr[i])):
                #cell = np.split(arr[i][a], 9)
            #count_white = np.sum(arr2[j] > 0)
            #count_black = np.sum(arr2[j] == 0)
            arr2[j] = arr2[j][5:-5 ,8:-8]
            #if count_black > count_white:
                #arr2[j] = 255 - arr2[j]
            arr2[j] = cv2.copyMakeBorder(arr2[j], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            #else:
                #arr2[j] = cv2.copyMakeBorder(arr2[j], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            value = re.sub('\x0c','',pt.image_to_string(arr2[j], config='--psm 6 --oem 1 -c tessedit_char_whitelist=123456789'))
            if value == '':
                value = '0\n'
            d.append(value)
            #d.encode("ascii", errors="ignore").decode()
            #d.remove('\x0c')
            #cv2.imwrite(f'imdt\{z}.png', arr2[j])
            #print(f'imdt\{z}.png' + 'stored')
            #z =z+1
                
'''
name of the input image file (with extension) located in input folder
'''
image_name = 'img9.jpg'


img = cv2.imread(f'input/{image_name}')
print('imported the image...')
h,w,ch = img.shape
if h>= w:
    img = img[ 0:-1,0:-1 ] ##crop
else:
    img = img[ 0:-1,0:-1 ] ##crop
puzzle = get_puzzle(img)
print('applied preprocessing...')
cv2.imshow('processed image', puzzle)
warp = warp_img(puzzle)
print('isolatd the puzzle area...')
#d = pt.image_to_string(warp, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
#t = threading.Thread(target=animate)
#t.start()
the_process = threading.Thread(name='break_parts_process', target=break_parts, args =[warp])
the_process.start()
while the_process.isAlive():
    animated_loading()
with open(f'out_{image_name}.txt', "w") as file:
    for num in d:
        file.write(num)
file.close()
the_process.join()
print('\nDone.')
t2 = time.time()
print(f"whole process took a total of {t2-t1} time")
time.sleep(0.5)

cv2.imshow('out', warp)
cv2.waitKey(0)
cv2.destroyAllWindows()

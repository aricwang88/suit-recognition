##This program isolates the suit and saves the isolated images

# Import necessary packages
import cv2
import numpy as np
import time
import Cards
import os

img_path = os.path.dirname(os.path.abspath(__file__)) + '/Training_Set/'

IM_WIDTH = 1280
IM_HEIGHT = 720

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# Initialize USB camera
cap = cv2.VideoCapture(0)


for Name in ['Spades','Diamonds','Clubs','Hearts']:

    filename = Name + '.jpg'

    print('Press "p" to take a picture of ' + filename)
    # Press 'p' to take a picture
    while(True):
        ret, frame = cap.read()
        cv2.imshow("Card",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            image = frame
            break

    # Pre-process image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    retval, thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)

    # Find contours and sort them by size
    dummy,cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

    # Assume largest contour is the card. If there are no contours, print an error
    flag = 0
    image2 = image.copy()

    if len(cnts) == 0:
        print('No contours found!')
        quit()

    card = cnts[0]

    # Approximate the corner points of the card
    peri = cv2.arcLength(card,True)
    approx = cv2.approxPolyDP(card,0.01*peri,True)
    pts = np.float32(approx)

    x,y,w,h = cv2.boundingRect(card)

    # Flatten the card and convert it to 200x300
    warp = Cards.flattener(image,pts,w,h)

    # Grab corner of card image, zoom, and threshold
    corner = warp[0:84, 0:32]
    #corner_gray = cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)
    corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
    corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
    retval, corner_thresh = cv2.threshold(corner_blur, 155, 255, cv2. THRESH_BINARY_INV)

    suit = corner_thresh[186:336, 0:128] # Grabs portion of image that shows suit
    dummy, suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    suit_cnts = sorted(suit_cnts, key=cv2.contourArea,reverse=True)
    x,y,w,h = cv2.boundingRect(suit_cnts[0])
    suit_roi = suit[y:y+h, x:x+w]
    suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
    final_img = suit_sized

    cv2.imshow("Image",final_img)

    # Save image
    print('Press "c" to continue.')
    key = cv2.waitKey(0) & 0xFF
    if key == ord('c'):
        cv2.imwrite(img_path+filename,final_img)


cv2.destroyAllWindows()
camera.close()

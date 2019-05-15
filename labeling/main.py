import cv2
import numpy as np
from utilities import *
from segmentation import *
import os

# write the filenames of images in images/ to images.txt
filenames_file = open("images.txt", "w")
for filename in os.listdir("images/"):
    filenames_file.write("{}\n".format(filename))
filenames_file.close()

# iterate through every image
filenames_file = open("images.txt", "r")
for filename in filenames_file.readlines():
    print("\n\n\n\n\n{}".format(filename))
    filename = filename[:-5] # remove .png and end of line
    image = cv2.imread("images/{}.jpg".format(filename))

    # find possible bounding boxes in image with segmentation algorithm 
    bboxes = segment(image)

    drawing = False # true if mouse is pressed
    ix,iy = -1,-1
    jx,jy = -1,-1

    # mouse callback function
    def draw_bbox(event,x,y,flags,param):
        global ix,iy,jx,jy,drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.rectangle(temp_image,(ix,iy),(x,y),(0,255,0),1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            jx,jy = x,y
            save_bbox(filename, image.shape, min(jx,ix), min(jy,iy), abs(jx-ix), abs(jy-iy))
            cv2.rectangle(image,(ix,iy),(jx,jy),(0,255,0),1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            delete_last_bbox(filename)
            cv2.rectangle(image,(ix,iy),(jx,jy),(0,0,255),1)

    # empty content of labels file
    labels_file = open("labels/{}.txt".format(filename), "w")
    labels_file.write("")
    labels_file.close()

    # propose bounding boxes and let user decide to discard or keep
    print("\nPress \'1\' to save, \'0\' to discard bounding box. Press \'Enter\' to continue to manual selection.")
    image = bbox_selection(filename, image, bboxes)
    temp_image = image.copy()

    # let user draw more bounding boxes
    print("\nDraw additionnal bounding boxes mith the mouse, right click to delete last bounding box. Press \'Enter\' to continue to next image.")
    cv2.setMouseCallback('image',draw_bbox)
    while(1):
        if drawing == True:
            cv2.imshow('image', temp_image)
        else:
            cv2.imshow('image', image)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break

print("\nBounding boxes saved.")

filenames_file.close()
cv2.destroyAllWindows()

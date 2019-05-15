import cv2
import numpy as np
from segmentation import *
import os

# iterate through every image
for filename in os.listdir("images/"):
    image = cv2.imread("images/{}".format(filename))
    image_seg, bboxes = segment(image)
    for bbox in bboxes:
        x,y,w,h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imwrite("results/{}_detected.jpg".format(filename[:-4]), image)
    cv2.imwrite("results/{}_segmented.jpg".format(filename[:-4]), image_seg)

import cv2
import numpy as np

def bbox_selection(filename, image, bboxes):
    if len(bboxes) == 0:
        cv2.imshow('image', image)
        return image
    for (x,y,w,h) in bboxes:
        while (1):
            temp_image = image.copy()
            cv2.rectangle(temp_image,(x,y),(x+w, y+h),(0,255,0),2)
            cv2.imshow('image',temp_image)
            # wait for keypress
            k = cv2.waitKey(1)
            if k == 49: # add bounding box to label file and to image
                save_bbox(filename, image.shape, x, y, w, h)
                cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0),1)
                cv2.rectangle(temp_image,(x,y),(x+w, y+h),(0,255,0),2)
                break
            elif k == 48: # discard bounding bbox
                break
            elif k == 13: # continue to next image
                return image
    return image

def save_bbox(filename, image_shape, x, y, w, h):
    H, W, _ = image_shape
    # handle borders
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x+w > W:
        w = W-x
    if y+h > H:
        h = H-y
    labels_file = open("labels/{}.txt".format(filename), "a")
    # convert to YOLO-format
    object_class = 0
    relative_center_x = (x + w/2)/W
    relative_center_y = (y + h/2)/H
    relative_width = w/W
    relative_height = h/H
    labels_file.write("{} {} {} {} {}\n".format(object_class, relative_center_x, relative_center_y, relative_width, relative_height))
    labels_file.close()

def delete_last_bbox(filename):
    labels_file = open("labels/{}.txt".format(filename), "r")
    lines = labels_file.readlines()[:-1]
    labels_file.close()
    labels_file = open("labels/{}.txt".format(filename), "w")
    for line in lines:
        labels_file.write(line)
    labels_file.close()

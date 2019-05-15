import cv2
import numpy as np

def hsv_filter(image, lower_color, upper_color, filter):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if filter == "median":
        image_hsv = cv2.medianBlur(image_hsv, 9)
    # if the color jumps from 255 to 0, as is the case for red
    if lower_color[0] > upper_color[0]:
        # left side of color spectrum
        mask = cv2.inRange(image_hsv, np.array([0,lower_color[1],lower_color[2]]), np.array([upper_color[0],upper_color[1],upper_color[2]]))
        res1 = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
        # right side of color spectrum
        mask = cv2.inRange(image_hsv, np.array([lower_color[0],lower_color[1],lower_color[2]]), np.array([255,upper_color[1],upper_color[2]]))
        res2 = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
        # combine left and right side of color spectrum
        res = cv2.bitwise_or(res1, res2)
    else:
        mask = cv2.inRange(image_hsv, np.array([lower_color[0],lower_color[1],lower_color[2]]), np.array([upper_color[0],upper_color[1],upper_color[2]]))
        res = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    return res

def segment(image):
    # filter image in hsv-domain
    image_hsv = hsv_filter(image, [170,120,130], [6,255,255], "median")

    # canny edge detector on s- and v-channels of hsv color space
    image_cs = cv2.Canny(image_hsv[:,:,1], 50, 100)
    image_cv = cv2.Canny(image_hsv[:,:,2], 100, 500)
    image_c = cv2.bitwise_or(image_cs, image_cv)

    # closing on result of canny edge detector to close the contours
    kernel = np.ones((13,13), np.uint8)
    image_c = cv2.morphologyEx(image_c, cv2.MORPH_CLOSE, kernel)

    # find contours on result of canny edge detector
    _, contours, [hierarchies] = cv2.findContours(image_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # assign hierarchy level (0, 1 or 2) to every contour, 2 is outer contours, 1 is inner contours and 0 are the innermost or smallest contours
    index = 0
    hierarchy_levels = [0]*len(contours)
    children = []
    while index != -1:
        # exclude small outer contours
        if cv2.contourArea(contours[index]) > 200:
            hierarchy_levels[index] = 2
            # add first children
            if hierarchies[index][2] != -1:
                children.append(hierarchies[index][2])
        index = hierarchies[index][0]
    for index in children:
        while index != -1:
            hierarchy_levels[index] = 1
            index = hierarchies[index][0]

    # create array of minenclosingcircle for contours in level 2 to find possible clusters of strawberries
    contour_radius = np.empty((0),dtype = int)
    contour_radius_dict = {}
    for index, level in enumerate(hierarchy_levels):
        if (level == 2):
            x, y, w, h = cv2.boundingRect(contours[index])
            # ignore strawberries at edge of image, they cause trouble because their contours is not closed
            if (x != 0) and (y != 0) and (x+w != image.shape[1]) and (y+h != image.shape[0]):
                (x,y), radius = cv2.minEnclosingCircle(contours[index])
                contour_radius_dict[radius] = index + 1
                contour_radius = np.append(contour_radius, radius)

    # find possible clusters of strawberries by finding the contours with abnormaly large enclosing circles
    data_mean, data_std = np.mean(contour_radius), np.std(contour_radius)
    outliers = [x for x in contour_radius if x > data_mean + 0.5*data_std]
    outlier_index = [contour_radius_dict[j] for j in outliers]

    # get bounding boxes for every contour of interest and return them
    bboxes = []
    image_output = image.copy()
    for index, (contour, level) in enumerate(zip(contours, hierarchy_levels), 1):
        if level == 2:
            cv2.drawContours(image_output, [contour], 0, (255,0,0), 2)
            # if the contour is a possible strawberry cluster, find bounding boxes of children
            if index in outlier_index:
                for index_child, (contour, hierarchy) in enumerate(zip(contours, hierarchies), 1):
                    if hierarchy[3]+1 == index and hierarchy_levels[index_child-1] == 1:
                        x,y,w,h = cv2.boundingRect(contour)
                        w = 1.2*w
                        h = 1.2*h
                        x -= 0.1*w
                        y -= 0.1*h
                        bboxes.append([int(x),int(y),int(w),int(h)])
                        cv2.drawContours(image_output, [contour], 0, (0,255,0), 2)
            else:
                x,y,w,h = cv2.boundingRect(contour)
                w = 1.2*w
                h = 1.2*h
                x -= 0.1*w
                y -= 0.1*h
                bboxes.append([int(x),int(y),int(w),int(h)])
    return image_output, bboxes

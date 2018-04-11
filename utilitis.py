import functools
import os

import cv2
import numpy as np
from imutils import face_utils


def findCircles(image):
    ''' This function finds circles in the given image and returns a list of them'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 21)
    # detect circles in the image
    circles = cv2.HoughCircles(gray,  # processed image in grayscale
                               cv2.HOUGH_GRADIENT,  # the only method in CV2 for detection
                               1.1,  # The inverse ratio of resolution.
                               5,  # Minimal distance between circle's centers
                               param1=80,  # Upper threshold for the internal Canny edge detector.
                               param2=10,  # Threshold for center detection.
                               minRadius=0, maxRadius=100)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        print("find {} circles".format(len(circles)))
    return circles


def decoSaveImage(func):
    names = ['leftEye{}.jpg', 'rightEye{}.jpg']
    func._counter = 0
    path = r'C:\Users\sberlab6\PycharmProjects\EyeTrackingForSB\data'

    @functools.wraps(func)
    def inner(*args, **kwargs):
        img, tlcorner = func(*args, **kwargs)
        filename = os.path.join(path, (names[func._counter % 2]).format(func._counter // 2))
        func._counter += 1
        cv2.imwrite(filename, img)
        print("{} is saved".format(filename))
        return img, tlcorner

    return inner


@decoSaveImage
def cropRectImage(image, points_list):
    '''
    This function crops a rectangle around ))) a given list of points with overfit and returns cropped image and
    top left corner coordinates
     '''
    min_x = min(points_list, key=lambda item: item[0])[0]
    min_y = min(points_list, key=lambda item: item[1])[1]
    max_x = max(points_list, key=lambda item: item[0])[0]
    max_y = max(points_list, key=lambda item: item[1])[1]
    overfit = 5
    # print(min_x,min_y,max_x,max_y)
    croped_image = image[min_y - overfit: max_y + overfit, min_x - overfit: max_x + overfit, ]
    # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 128, 255), 1)
    return croped_image, (min_x - overfit, min_y - overfit)


def drawCircles(image, circles, offset=(0, 0)):
    ''' this function draws circles on the image with given coordinates plus offset'''
    # loop over the (x, y) coordinates and radius of the circles
    if circles:  # remove after all
        print(circles)  # !!!!! Тут проблема!!!!
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            x += offset[0]
            y += offset[1]
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

    return image


(rLeft, rRight) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lLeft, lRight) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]


def iris_replace(circle, eye_landmarks, offset=(0, 0)):
    x, y, r = circle
    x += offset[0]
    y += offset[1]
    center_x = (eye_landmarks[0][0] + eye_landmarks[3][0]) / 2
    center_y = (eye_landmarks[0][1] + eye_landmarks[3][1]) / 2
    return (x - center_x, y - center_y)


def filter_circles(circles, eye_landmarks, offset=(0, 0)):
    '''
    Initially, for every circle it finds its displace and then finds the smallest one.
    after that it finds out its index and returns a circle with this index.
    :param circles: list of circles
    :param eye_landmarks: list of landmarks
    :param offset: offset of circles
    :return: circle that is closer to the center of an eye
    '''
    if circles is not None:
        displaces = [np.linalg.norm(np.array(iris_replace(c, eye_landmarks, offset))) for c in circles]
        return circles[displaces.index(min(displaces))]


def all_in_one_pocessing(frame, shape):
    leye, lshift = cropRectImage(frame, shape[lLeft:lRight])
    reye, rshift = cropRectImage(frame, shape[rLeft:rRight])

    liris = findCircles(leye)
    riris = findCircles(reye)

    li = filter_circles(liris, shape[lLeft:lRight], lshift)
    ri = filter_circles(riris, shape[rLeft:rRight], rshift)
    print("Left eye {} and Right eye {}".format(li, ri))
    ldisplace, rdisplace = None, None

    # ужасный костыль, надо исправить.
    if li is not None:
        ldisplace = iris_replace(li, shape[lLeft:lRight], lshift)
        liris = [li]
    else:
        liris = None

    if ri is not None:
        rdisplace = iris_replace(ri, shape[rLeft:rRight], rshift)
        riris = [ri]
    else:
        riris = None

    drawCircles(frame, liris, lshift)
    drawCircles(frame, riris, rshift)

    for (x, y) in shape[rLeft:rRight]:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    for (x, y) in shape[lLeft:lRight]:
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    return (ldisplace, rdisplace)
    pass

import functools
import os

import cv2
import dlib
import numpy as np
from imutils import face_utils

import templateMethod as tm


def findCircles(image):
    ''' This function finds circles in the given image and returns a list of them'''
    expected_radius = int(image.shape[1] / 2)  # assume an iris is  one third of an eye width plus 25%

    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    # image = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=0.35)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # 27 - best result
    # detect circles in the image
    # circles = cv2.HoughCircles(gray,  # processed image in grayscale
    #                            cv2.HOUGH_GRADIENT,  # the only method in CV2 for detection
    #                            3,  # The inverse ratio of resolution.
    #                            image.shape[1]//2,  # Minimal distance between circle's centers
    #                            param1=150,  # Upper threshold for the internal Canny edge detector.
    #                            param2=40,  # Threshold for center detection.
    #                            minRadius=expected_radius//2, maxRadius=expected_radius)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               1.8,
                               image.shape[1] // 2,
                               param1=60,
                               param2=20,
                               minRadius=2 * expected_radius // 4,
                               maxRadius=expected_radius)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        # circles = np.round(circles[0, :]).astype("int")
        pass
        #print('\n')
        #print(circles)
        # print("find {} circle(s)".format(len(circles)))
        return circles[0, 0]


def decoSaveImage(func):
    names = ['leftEye{}.jpg', 'rightEye{}.jpg']
    func._counter = 0
    path = '.\\data'
    func.args = []
    func.kwargs = None

    @functools.wraps(func)
    def inner(*args, **kwargs):
        # if func._counter // 2 == 0:
        #     func.args.append(args)
        #     func.kwargs = kwargs
        img, tlcorner = func(*args, **kwargs)
        filename = os.path.join(path, (names[func._counter % 2]).format(func._counter // 2))
        func._counter += 1
        cv2.imwrite(filename, img)
        #print("{} is saved".format(filename))
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
    overfit = 50
    # print(min_x,min_y,max_x,max_y)
    croped_image = image[min_y - overfit: max_y + overfit, min_x - overfit: max_x + overfit, ]
    # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 128, 255), 1)
    return croped_image, (min_x - overfit, min_y - overfit)


def drawCircles(image, circles, offset=(0, 0)):
    ''' this function draws circles on the image with given coordinates plus offset'''
    # loop over the (x, y) coordinates and radius of the circles
    if circles:  # remove after all
        #print(circles)  # !!!!! Тут проблема!!!!
        for c in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            x = int(c[0] + offset[0])
            y = int(c[1] + offset[1])
            r = int(c[2])
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

    return image


(rLeft, rRight) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lLeft, lRight) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]


def iris_replace(circle, eye_landmarks, offset=(0, 0)):
    x, y, r = circle[0], circle[1], circle[2]
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
        displaces = [np.linalg.norm(np.array(iris_replace(c, eye_landmarks, offset) / c[2])) for c in circles]
        return [circles[displaces.index(min(displaces))]]


def decoSaveDataBeetwenCalls(func):
    func.counter = 0
    func.dataToStore = []

    @functools.wraps(func)
    def inner(*args, **kwargs):
        if func.counter < 2:
            func.dataToStore.append(args[1])
        #print('\nStrored Data {}\n'.format(func.dataToStore[func.counter % 2]))
        result = func(args[0], func.dataToStore[func.counter % 2], **kwargs)
        func.dataToStore[func.counter % 2] = result
        func.counter += 1
        return np.array([result])
        pass

    return inner


@decoSaveDataBeetwenCalls
def filter_circles_dark_point(circles, dark_point, offset=(0, 0)):
    if circles is not None:
        circles = np.array(circles)
        temp = dark_point
        distances = circles - temp  # get dist to every circle dx, dy
        distances_len = np.array([d[0] ** 2 + d[1] ** 2 + d[2] ** 2 for d in distances])  # get distance lenght
        min_dist_index = np.argmin(distances_len)  # find median value
        return [circles[min_dist_index]]  # leave only circle with maximum radius
    else:
        return dark_point


def all_in_one_pocessing(frame, shape):
    leye, lshift = cropRectImage(frame, shape[lLeft:lRight])
    reye, rshift = cropRectImage(frame, shape[rLeft:rRight])

    liris = findCircles(leye)
    riris = findCircles(reye)

    li = filter_circles(liris, shape[lLeft:lRight], lshift)
    ri = filter_circles(riris, shape[rLeft:rRight], rshift)
    #print("Left eye {} and Right eye {}".format(li[0], ri[0]))
    ldisplace, rdisplace = None, None

    # ужасный костыль, надо исправить.
    if li[0] is not None:
        ldisplace = iris_replace(li[0], shape[lLeft:lRight], lshift)
        liris = [li[0]]
    else:
        liris = None

    if ri[0] is not None:
        rdisplace = iris_replace(ri[0], shape[rLeft:rRight], rshift)
        riris = [ri[0]]
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


def find_darkest_point(image):
    #image = cv2.bilateralFilter(image, 13, 25, 25)
    local_image = image.copy()
    # gray = cv2.cvtColor(local_image, cv2.COLOR_BGR2GRAY)
    Gimage = cv2.GaussianBlur(local_image, (7, 7), 3, 0)
    Center_min = np.argwhere(Gimage == np.min(Gimage))
    a_c = np.mean(Center_min, axis=0)
    return np.array([[a_c[1], a_c[0], 15]])


def all_in_one_processing_with_G(frame, shape):
    leye, lshift = cropRectImage(frame, shape[lLeft:lRight])
    reye, rshift = cropRectImage(frame, shape[rLeft:rRight])

    li = find_darkest_point(leye)
    ri = find_darkest_point(reye)
    # print("Left eye {} and Right eye {}".format(li, ri))
    ldisplace, rdisplace = None, None

    # ужасный костыль, надо исправить.
    if li is not None:
        ldisplace = iris_replace(li, shape[lLeft:lRight], lshift)

    if ri is not None:
        rdisplace = iris_replace(ri, shape[rLeft:rRight], rshift)

    li = tuple(int(x) for x in li)
    ri = tuple(int(x) for x in ri)

    drawCircles(frame, [li], lshift)
    drawCircles(frame, [ri], rshift)

    for (x, y) in shape[rLeft:rRight]:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    for (x, y) in shape[lLeft:lRight]:
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    return (ldisplace, rdisplace)
    pass


def all_in_one_processing_mixed(frame, shape):
    ldisplace, rdisplace = None, None

    leye, lshift = cropRectImage(frame, shape[lLeft:lRight])
    reye, rshift = cropRectImage(frame, shape[rLeft:rRight])

    liris = findCircles(leye)
    riris = findCircles(reye)

    dark_left_eye = find_darkest_point(leye)
    dark_right_eye = find_darkest_point(reye)

    li = filter_circles_dark_point(liris, dark_left_eye, lshift)
    ri = filter_circles_dark_point(riris, dark_right_eye, rshift)
    #print("Left eye {} and Right eye {}".format(li[0], ri[0]))
    # ужасный костыль, надо исправить.
    if li[0] is not None:
        # ldisplace = iris_replace(li[0], shape[lLeft:lRight], lshift)
        liris = li
    else:
        liris = None

    if ri[0] is not None:
        # rdisplace = iris_replace(ri[0], shape[rLeft:rRight], rshift)
        riris = ri
    else:
        riris = None

    drawCircles(frame, [liris], lshift)
    drawCircles(frame, [riris], rshift)

    for (x, y) in shape[rLeft:rRight]:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    for (x, y) in shape[lLeft:lRight]:
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    return (ldisplace, rdisplace)
    pass


def all_in_one_processing_corr(frame, shape):
    ldisplace, rdisplace = None, None

    leye, lshift = cropRectImage(frame, shape[lLeft:lRight])
    reye, rshift = cropRectImage(frame, shape[rLeft:rRight])
    expected_radius = 21
    liris = tm.GetEyeCenterByTemplate(leye, expected_radius)
    # liris += (expected_radius,)
    riris = tm.GetEyeCenterByTemplate(reye, expected_radius)
    # riris += (expected_radius,)
    #print(liris, riris)
    drawCircles(frame, [liris], lshift)
    drawCircles(frame, [riris], rshift)

    # ldisplace = iris_replace(liris, shape[lLeft:lRight], lshift)
    # rdisplace = iris_replace(riris, shape[rLeft:rRight], rshift)

    ldisplace = [a + b for a, b in zip(liris[:2], lshift)] + list(liris[2:])
    rdisplace = [a + b for a, b in zip(riris[:2], rshift)] + list(riris[2:])
    return (ldisplace, rdisplace)


def all_in_one_processing_corr_with_sq(frame, shape):
    ldisplace, rdisplace = None, None

    leye, lshift = cropSquareImage(frame, shape[lLeft:lRight])
    reye, rshift = cropSquareImage(frame, shape[rLeft:rRight])

    liris = findCircles(leye)
    riris = findCircles(reye)

    li = filter_circles(liris, shape[lLeft:lRight], lshift)
    ri = filter_circles(riris, shape[rLeft:rRight], rshift)
    print("Left eye {} and Right eye {}".format(li[0], ri[0]))
    ldisplace, rdisplace = None, None

    # ужасный костыль, надо исправить.
    if li[0] is not None:
        ldisplace = iris_replace(li[0], shape[lLeft:lRight], lshift)
        liris = [li[0]]
    else:
        liris = None

    if ri[0] is not None:
        rdisplace = iris_replace(ri[0], shape[rLeft:rRight], rshift)
        riris = [ri[0]]
    else:
        riris = None

    drawCircles(frame, liris, lshift)
    drawCircles(frame, riris, rshift)

    for (x, y) in shape[rLeft:rRight]:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    for (x, y) in shape[lLeft:lRight]:
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    return (ldisplace, rdisplace)


def cropSquareImage(image, points_list):
    '''
    This function crops a rectangle around ))) a given list of points with overfit and returns cropped image and
    top left corner coordinates
     '''
    min_x = min(points_list, key=lambda item: item[0])[0]
    min_y = min(points_list, key=lambda item: item[1])[1]
    max_x = max(points_list, key=lambda item: item[0])[0]
    max_y = max(points_list, key=lambda item: item[1])[1]
    overfit = 50
    # print(min_x,min_y,max_x,max_y)
    cropped_image = image[min_y - overfit: max_y + overfit, min_x - overfit: max_x + overfit, ]
    radius = (max_x - min_x) // 4
    centre = tm.GetEyeCenterByTemplate(cropped_image,
                                       radius)  # it is because an iris's radius is aproximately one third of eye's width
    cv2.circle(cropped_image, centre[:2], centre[2], (127, 35, 68), 1)
    centre = (centre[0] + (min_x - overfit),
              centre[1] + (min_y - overfit))  # corret position to initial top left corner of an image
    square_cropped_image = image[centre[1] - radius:centre[1] + radius, centre[0] - radius:centre[0] + radius, ]
    return square_cropped_image, (centre[0] - radius, centre[1] - radius)
    pass


if __name__ == '__main__':
    for i in range(0, 568,1):
        print(i)
        image = cv2.imread('.\\data\\LeftEye{}.jpg'.format(i))
        liris = findCircles(image)
        dark_left_eye = find_darkest_point(image)
        li = filter_circles_dark_point(liris, dark_left_eye)
        drawCircles(image, li)
        print(dark_left_eye)
        print(li)

        cv2.imshow("test", image)
        cv2.waitKey()

import datetime
import threading
import functools
import time
import os
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream

from Object import CoreObject


class CamRecoder(CoreObject):

    def __init__(self, screen, filename=None):
        #super.__init__()
        self.cap = VideoStream(0)
        self.cap.start()

        self.filename = filename

        self.recordingThread = None
        pass

    def start_record(self):

        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.daemon = True
        self.recordingThread.start()

    def stop_record(self):


        if self.recordingThread:
            self.recordingThread.stop()

    def update(self):
        # I'm going to use this method to update an eye gaze marker on the screen further
        pass

    def draw(self, surface):
        # I'm going to use this method to draw an eye gaze marker on the screen further
        pass

    def finish(self):
        self.stop_record()
        self.recordingThread.join()
        self.cap.stop()


class RecordingThread(threading.Thread):
    def __init__(self, name, camera, filename=None):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        if filename is None:
            filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".avi"

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(filename, fourcc, 5.0, (800, 600), isColor=True)

    def run(self):
        print("Recording starts.....")
        (rLeft, rRight) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (lLeft, lRight) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        begin_time = time.time()
        while self.isRunning:
            frame = self.cap.read()
            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leye, lshift = cropRectImage(frame, shape[lLeft:lRight])
                reye, rshift = cropRectImage(frame, shape[rLeft:rRight])
                liris = findCircles(leye)
                riris = findCircles(reye)
                drawCircles(frame,liris,lshift)
                drawCircles(frame,riris, rshift)
                for (x, y) in shape[rLeft:rRight]:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                for (x, y) in shape[lLeft:lRight]:
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
            self.out.write(frame)
            end_time = time.time()
            #print(" iteration time is {}".format(end_time - begin_time))
            begin_time=end_time
        print('Recording is finished.....')
        self.out.release()

    def stop(self):

       self.isRunning = False



def findCircles(image):
    ''' This function finds circles in the given image and returns a list of them'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # detect circles in the image
    circles = cv2.HoughCircles(gray,  # processed image in grayscale
                               cv2.HOUGH_GRADIENT,  # the only method in CV2 for detection
                               1,  # The inverse ratio of resolution.
                               25,  # Minimal distance between circle's centers
                               param1=50,  # Upper threshold for the internal Canny edge detector.
                               param2=1,  # Threshold for center detection.
                               minRadius=8, maxRadius=14)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        print("find {} circles".format(len(circles)))
    return circles

def decoSaveImage(func):
    names = ['leftEye{}.jpg','rightEye{}.jpg']
    func._counter = 0
    path = r'C:\Users\E7470\PycharmProjects\EyeTrackingForSB\data'
    @functools.wraps(func)
    def inner(*args, **kwargs):
        img, tlcorner = func(*args,**kwargs)
        filename = os.path.join(path,(names[func._counter % 2]).format(func._counter //2))
        func._counter +=1
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
    print(min_x,min_y,max_x,max_y)
    croped_image = image[ min_y-overfit: max_y+overfit, min_x-overfit: max_x+overfit,]
    #cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 128, 255), 1)
    return croped_image, (min_x-overfit,min_y-overfit)

def drawCircles(image, circles, offset = (0,0)):
    ''' this function draws circles on the image with given coordinates plus offset'''
    # loop over the (x, y) coordinates and radius of the circles
    if circles is not None:
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            x+=offset[0]
            y+=offset[1]
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

    return image


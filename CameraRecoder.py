import datetime
import threading

import cv2
import dlib
import imutils
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


        if self.recordingThread != None:
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
        self.out = cv2.VideoWriter(filename, fourcc, 30.0, (800, 600), isColor=True)

    def run(self):
        print("Recording starts.....")
        while self.isRunning:
            frame = self.cap.read()
            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            self.out.write(frame)
        print('Recording is finished.....')
        self.out.release()

    def stop(self):

       self.isRunning = False


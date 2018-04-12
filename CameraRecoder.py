import datetime
import threading

import dlib
import pygame
from imutils.video import FileVideoStream

from Object import CoreObject
from calibration import *
from utilitis import *

calibration_data['TLEyePos'] = Point(16.5, +30.0)
calibration_data['TREyePos'] = Point(-17.0, +30.5)
calibration_data['BLEyePos'] = Point(12.5, -2.5)
calibration_data['BREyePos'] = Point(-4.5, -3.0)

calibrator = Calibrator((1920, 1080), calibration_data)

class CamRecoder(CoreObject):

    def __init__(self, screen, x, y, r, color=pygame.Color('red'), filename=None):
        CoreObject.__init__(self, x - r, y - r, 2 * r, 2 * r)
        self.on_screen = screen  # Is used just to know the canvas size
        self.color = color
        self.radius = r
        self.x = x
        self.y = y

        self.cap = FileVideoStream('.\data\SAMPLE_VIDEO3.mp4')
        # self.cap = VideoStream(0)
        self.cap.start()

        self.filename = filename

        self.recordingThread = None

        self.calibrator = Calibrator(screen.get_size(), calibration_data)

        pass

    def start_record(self):

        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.daemon = True
        self.recordingThread.start()
        pass

    def stop_record(self):
        if self.recordingThread:
            self.recordingThread.stop()
        pass

    def update(self):
        # I'm going to use this method to update an eye gaze marker on the screen further
        eye_pos = self.recordingThread.positions
        # print('update func {}'.format(eye_pos))
        if eye_pos and eye_pos[0]:
            center = calibrator.translate(eye_pos[0])
            # print('update func {}'.format(center))
            self.x, self.y = tuple(int(x) for x in center)
            print('New coordinates are {}, {}'.format(self.x, self.y))

        pass

    def draw(self, surface):
        # I'm going to use this method to draw an eye gaze marker on the screen further
        pygame.draw.circle(surface,
                           pygame.Color('yellow'),
                           (self.x, self.y),
                           20)
        pass

    def finish(self):
        self.stop_record()
        self.recordingThread.join()
        self.cap.stop()
        pass


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
        width = int(self.cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Thi video file frame size is {} x {} pxls".format(width, height))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter(filename, fourcc, 15.0, (width, height), isColor=True)
        self.stateTwoEyes = None
        pass

    def run(self):
        print("Recording starts.....")
        while self.isRunning:
            frame = self.cap.read()
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)#!!! УБРАТЬ !!!!
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                self.stateTwoEyes = all_in_one_pocessing_with_G(frame, shape)

            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)#!!! УБРАТЬ !!!!
            self.out.write(frame)

        print('Recording is finished.....')
        self.out.release()
        pass

    @property
    def positions(self):
        return self.stateTwoEyes
        pass

    def stop(self):
        self.isRunning = False
        pass

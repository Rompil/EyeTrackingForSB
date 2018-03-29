# import all necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import cv2
import time
import dlib

# construct the argument parser and parse arguments

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to landmark predictor")  # check a path to the predictor file
# # ap.add_argument("-r", "--picamera", type=int, default=-1)# I'm not going to run this code on the rasberry Pi board
# args = vars(ap.parse_args())

## initialize dlib's predictor and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor ...")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] camera sensor is warming up ... ")
vs = VideoStream(0)
vs.start()
time.sleep(2.0)
print("You are here ...")
# loop over the frames from the video stream
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()

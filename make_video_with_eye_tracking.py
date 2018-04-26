import datetime

import cv2
import numpy as np

from calibration import calibration_data, Calibrator, Point

calibration_data['TLEyePos'] = Point(-20, -7.5)
calibration_data['TREyePos'] = Point(7, -7.5)
calibration_data['BLEyePos'] = Point(-20, -20.0)
calibration_data['BREyePos'] = Point(7, -20.0)

cal = Calibrator((1024, 768), calibration_data)

path_to_file = "test.dat"


def get_points(file=path_to_file):
    eyes = []
    with open('C:\\Users\\sberlab6\\PycharmProjects\\EyeTrackingForSB\\test.dat', 'r') as f:
        for line in f:
            t = ''.join([c for c in line if c not in '(),']).split()
            t = [float(f) for f in t]
            eyes.append(t)
    return np.array(eyes)


def make_video():
    eyes = get_points(path_to_file)
    print(eyes.shape)
    width, height = 1024, 768  # video file frame size
    filename = 'eye_track_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".avi"
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(filename, fourcc, 15, (width, height), True)
    track_frame = np.zeros((height, width, 3), dtype='uint8')
    l, r = 2, 4
    prev_values = eyes[0]

    for gazes in eyes:
        # print(gazes)
        last_point = np.zeros((height, width, 3), dtype='uint8')
        start_point = cal.translate(tuple(prev_values[l:r].tolist()))
        end_point = cal.translate(tuple((gazes[l:r].tolist())))
        print(end_point)
        cv2.line(track_frame, start_point, end_point, (255, 255, 255), 1)
        cv2.circle(last_point, end_point, 5, (0, 0, 255), -1)
        result = track_frame + last_point
        prev_values = gazes
        cv2.imshow('test', result)
        cv2.waitKey(-1)
        # result = np.zeros((height, width, 3), dtype="uint8")
        # print(result.shape)
        out.write(result)
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    make_video()

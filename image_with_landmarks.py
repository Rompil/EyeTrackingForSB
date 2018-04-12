import dlib

from utilitis import *

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
name = ".\data\SAMPLE_VIDEO3_TR"
ext = ".JPG"
frame = cv2.imread(name + ext)
if frame is not None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    print('Image didn\'t load')
rects = detector(gray, 0)
for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    ld, rd = all_in_one_pocessing(frame, shape)
    print(ld, rd)
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), 3)

# frame = cv2.resize(frame, (int(0.25* width), int(0.25 * height)), interpolation=cv2.INTER_CUBIC)

cv2.imshow("Frame", frame)
cv2.waitKey(5)
full_name = name + '_with_landmarks' + ext
cv2.imwrite(full_name, frame)
print(full_name)

class Calibration():
    def __init__(self):
        pass

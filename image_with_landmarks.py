from utilitis import *


def cropEyeOnly(frame, shape):
    (lLeft, lRight) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    left_eye = shape[lLeft:lRight]
    rect = cv2.boundingRect(left_eye)
    x, y, w, h = rect
    croped = frame[y:y + h, x:x + w].copy()

    pts = left_eye - left_eye.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    cv2.imshow("croped.png", croped)
    cv2.imshow("mask.png", mask)
    cv2.imshow("dst.png", dst)
    cv2.imshow("dst2.png", dst2)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
name = ".\data\\Topleft"
ext = ".JPG"
frame = cv2.imread(name + ext)
if frame is not None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    print('Image didn\'t load')
rects = detector(gray, 0)

# frame = cv2.resize(frame, (int(0.25* width), int(0.25 * height)), interpolation=cv2.INTER_CUBIC)

for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    cropEyeOnly(frame, shape)
    ld, rd = all_in_one_processing_corr(frame, shape)
    print(ld, rd)
    # for (x, y) in shape:
    #     cv2.circle(frame, (x, y), 1, (0, 0, 255), 3)
    output = face_utils.visualize_facial_landmarks(frame, shape)
cv2.imshow("Frame", output)
cv2.waitKey(-1)
full_name = name + '_with_landmarks' + ext
cv2.imwrite(full_name, frame)

print(full_name)
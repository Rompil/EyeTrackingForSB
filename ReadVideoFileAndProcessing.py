import datetime

import dlib

from utilitis import *


def main():
    # cap = FileVideoStream("GOPR9183.MP4").start()
    cap = cv2.VideoCapture(".\\data\\20180417_163149.MP4")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #print("The video file frame size is {} x {} pxls".format(width, height))
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".avi"
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter()
    out.open(filename, fourcc, 24.0, (width, height), isColor=True)
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # !!!!! it is better to remove this line

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            counter += 1
            #print(counter)

            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                ld, rd = all_in_one_processing_corr(frame, shape)
                print(ld, rd)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # frame = cv2.resize(frame, (int(0.25* width), int(0.25 * height)), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(-1)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # !!!! and this too
            #out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

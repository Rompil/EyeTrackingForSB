'''
struct EyeTemplates
{
	Mat Positive, Negative;
};

EyeTemplates GetEyeTemplates(int eyeradius)
{
	int bigradius = sqrt(2.)*eyeradius+0.5;
	int xc = bigradius, yc = bigradius;
	EyeTemplates Templates;
	Templates.Positive = Mat::zeros(2 * bigradius, 2 * bigradius, CV_8U);
	Templates.Negative = Mat::zeros(2 * bigradius, 2 * bigradius, CV_8U);
	for (int i = 0; i < Templates.Positive.rows; ++i)
	{
		uchar* posline = Templates.Positive.ptr<uchar>(i);
		uchar* negline = Templates.Negative.ptr<uchar>(i);
		for (int j = 0; j < Templates.Positive.cols; ++j)
		{
			int dx = j - xc;
			int dy = i - yc;
			float dist = sqrt(dx*dx + dy * dy);
			if (dist < bigradius && dist > eyeradius)
				posline[j] = 1;
			if (dist < eyeradius)
				negline[j] = 1;
		}
	}
	return Templates;
}

Point GetEyeCenterByTemplate(Mat eye, int eyeradius)
{
	EyeTemplates Templates = GetEyeTemplates(eyeradius);
	Mat PosRes, NegRes;
	matchTemplate(eye, Templates.Positive, PosRes, CV_TM_CCORR);
	matchTemplate(eye, Templates.Negative, NegRes, CV_TM_CCORR);
	Mat Res = PosRes - NegRes;

	Point Shift(Templates.Positive.cols / 2, Templates.Positive.cols / 2);
	double vmax,vmin;
	Point pmax,pmin;

	minMaxLoc(Res, &vmin, &vmax, &pmin, &pmax);
	pmax += Shift;
	//pmin += Shift;

	return pmax;
}
'''
import cv2
import numpy as np


class EyeTemplates:

    def __init__(self, Negative=None, Positive=None):
        self.negative = Negative
        self.positive = Positive


def GetEyeTemplates(eyeradius):
    bigradius = int(np.sqrt(2.) * eyeradius + 0.5)
    templates = EyeTemplates(np.zeros((2 * bigradius, 2 * bigradius, 1), np.uint8),
                             np.zeros((2 * bigradius, 2 * bigradius, 1), np.uint8))
    cv2.circle(templates.positive, (bigradius, bigradius), bigradius, 255, -1)
    cv2.circle(templates.positive, (bigradius, bigradius), eyeradius, 0, -1)
    cv2.circle(templates.negative, (bigradius, bigradius), eyeradius, 255, -1)
    return templates


def GetEyeCenterByTemplate(eye, eyeradius):
    cv2.normalize(eye, eye, 0, 255, cv2.NORM_MINMAX)
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    r_list = [eyeradius - 1, eyeradius, eyeradius + 1]
    max_loc = []
    max_val = []
    i = 0
    for r in r_list:
        templates = GetEyeTemplates(r)
        PosRes = cv2.matchTemplate(eye, templates.positive, cv2.TM_CCORR)
        NegRes = cv2.matchTemplate(eye, templates.negative, cv2.TM_CCORR)
        Res = PosRes - NegRes
        Res = Res / (r ** 2)
        temp = cv2.minMaxLoc(Res)
        max_val.append(temp[1])
        max_loc.append(temp[3])
    max_index = np.argmax(max_val)
    w, h = templates.negative.shape[:2]
    maximum_location = (max_loc[max_index][0] + w // 2, max_loc[max_index][1] + h // 2)
    #print(maximum_location)
    return (*maximum_location, r_list[max_index])


if __name__ == '__main__':
    t = GetEyeTemplates(300)
    # cv2.imshow("positive",t.positive)
    # cv2.imshow("negative", t.negative)
    # plt.subplot(121), plt.imshow(t.positive, cmap='gray')
    # plt.title('Positive'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(t.negative, cmap='gray')
    # plt.title('Negative'), plt.xticks([]), plt.yticks([])

    # plt.show()

    image = cv2.imread('.\\data\\test_eye.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    point = GetEyeCenterByTemplate(image, 105)
    print(point)
    cv2.circle(image, point, 1, 200, 2)
    cv2.circle(image,
               (point[0], point[1]),
               105,
               255,
               4)
    cv2.imshow('Eye', image)
    cv2.waitKey()

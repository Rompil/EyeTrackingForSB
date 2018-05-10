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
int _refineSrchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    cv::Mat matWarp = cv::Mat::eye(2, 3, CV_32FC1);
    matWarp.at<float>(0,2) = ptResult.x;
    matWarp.at<float>(1,2) = ptResult.y;
    int number_of_iterations = 200;
    double termination_eps = 1e-10;

    cv::findTransformECC ( matTmpl,
                           mat,
                           matWarp,
                           MOTION_TRANSLATION,
                           TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                        number_of_iterations,
                                        termination_eps)
                            );
    ptResult.x = matWarp.at<float>(0,2);
    ptResult.y = matWarp.at<float>(1,2);
    return 0;
}

'''
import cv2
import numpy as np


def _refineSrchTemplate(mat, matTmpl, ptResult):
    '''

    :param mat:
    :param matTmpl:
    :param ptResult:
    :return:
    '''
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2], warp_matrix[1, 2] = ptResult[0], ptResult[1]
    # Specify the number of iterations.
    number_of_iterations = 1200;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-6;
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(matTmpl, mat, warp_matrix, warp_mode, criteria)
    except Exception:
        pass
    return warp_matrix[0, 2], warp_matrix[1, 2]


def subPixelAcc(image, location):
    A = np.array([[1, 1, -1, -1, 1],
                  [1, 0, -1, 0, 1],
                  [1, 1, -1, 1, 1],
                  [0, 1, 0, -1, 1],
                  [0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 1],
                  [1, 1, 1, -1, 1],
                  [1, 0, 1, 0, 1],
                  [1, 1, 1, 1, 1]])
    Y = image[location[0] - 1:location[0] + 2, location[1] - 1:location[1] + 2]
    Y = np.reshape(Y, (9, 1))
    aTa = np.dot(A.T, A)
    inv_aTa = np.linalg.inv(aTa)
    aTY = np.dot(A.T, Y)
    W = np.dot(inv_aTa, aTY)
    dx = -W[2] / (2 * W[0])
    dy = -W[3] / (2 * W[1])
    return (location[0] + dx, location[1] + dy)


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
    var = 1
    r_list = range(eyeradius - var, eyeradius + var + 1)
    max_loc = []
    max_val = []
    i = 0
    for r in r_list:
        # print(r)
        templates = GetEyeTemplates(r)
        PosRes = cv2.matchTemplate(eye, templates.positive, cv2.TM_CCORR)
        NegRes = cv2.matchTemplate(eye, templates.negative, cv2.TM_CCORR)
        Res = PosRes - NegRes
        Res = Res / (r ** 2)
        temp = cv2.minMaxLoc(Res)
        # float_location = _refineSrchTemplate(eye, templates.positive, temp[3])
        float_location = subPixelAcc(Res, temp[3])
        max_val.append(temp[1])
        max_loc.append(float_location)
    max_index = np.argmax(max_val)
    w, h = templates.negative.shape[:2]
    maximum_location = (max_loc[max_index][0] + w / 2, max_loc[max_index][1] + h / 2)

    # print('Max location ', maximum_location)
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

    image = cv2.imread('.\\data\\lefteye100.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    point = GetEyeCenterByTemplate(image, 21)

    print(point)
    ipoint = np.around(point).astype('int')
    cv2.circle(image, (ipoint[0], ipoint[1]), 1, 200, 2)
    cv2.circle(image,
               (ipoint[0], ipoint[1]),
               point[2],
               255,
               2)
    cv2.imshow('Eye', image)
    cv2.waitKey()

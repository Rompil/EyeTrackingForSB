import json
from glob import glob

import cv2
import numpy as np

import utilitis


class Tester():

    def __init__(self, folder):
        self.images = glob(folder + '\\*.jpg')
        pass

    def setMethod(self, method):
        self.method = method
        pass

    def __processImage(self, image_fn):
        img = cv2.imread(image_fn)
        data_file = open('{}.json'.format(image_fn[:-4]))
        data = json.load(data_file)

        def process_json_list(json_list):
            ldmks = [eval(s) for s in json_list]
            return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

        ldmks_iris = process_json_list(data['iris_2d'])
        eye_c = np.mean(ldmks_iris[:, :2], axis=0)
        # eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)

        method_result = self.method(img)
        method_result = method_result.tolist()
        return method_result, eye_c

    def run_test(self):
        result, etalon = [], []
        counter = 0
        for img in self.images[0:]:  # here shrink a list to process
            r, e = self.__processImage(img)
            result.append(r)
            etalon.append(e)
            print(counter)
            counter += 1
        self.result = result
        self.etalon = etalon
        return result, etalon

    def compareResults(self):
        diff = []
        for r, e in zip(self.result, self.etalon):
            r = np.array(r).flatten()
            d = [r[0] - e[0], r[1] - e[1]]
            diff.append(d)
        return diff
        pass

    def save_to_file(self, file_name):
        with open(file_name, 'w') as datafile:
            for r, e in zip(self.result, self.etalon):
                r = np.array(r).flatten()
                datafile.write('{}, {}, {}, {}\n'.format(r[0], r[1], e[0], e[1]))
def dummy_method(img):
    cx, cy = img.shape[:2]
    cx /= 2
    cy /= 2
    return cx, cy


if __name__ == '__main__':
    t = Tester('.\\imgs')
    t.setMethod(utilitis.find_darkest_point)
    # t.setMethod(utilitis.findCircles)
    t.run_test()
    r = t.compareResults()
    t.save_to_file('toRemove_darkest_point.csv')
    print(r)

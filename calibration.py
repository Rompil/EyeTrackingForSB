from collections import namedtuple

Point = namedtuple('Point', 'x y')
Screen_size = namedtuple('Screen_size', 'width height')

calibration_data = {}
calibration_data['TLEyePos'] = Point(10, 10)
calibration_data['TREyePos'] = Point(10, 10)
calibration_data['BLEyePos'] = Point(-10, -10)
calibration_data['BREyePos'] = Point(10, -10)


class Calibrator():
    def __init__(self, screen_size, cal_data=None):
        self.screen_size = Screen_size(*screen_size)
        if cal_data is not None:
            self.setCalData(cal_data)
        pass

    def setCalData(self, cal_data):
        self.xeyeleft = (cal_data['TLEyePos'].x + cal_data['BLEyePos'].x) / 2
        self.xyeyright = (cal_data['TREyePos'].x + cal_data['BREyePos'].x) / 2
        self.yeyetop = (cal_data['TLEyePos'].y + cal_data['TREyePos'].y) / 2
        self.yeyebottom = (cal_data['BLEyePos'].y + cal_data['BREyePos'].y) / 2
        pass

    def translate(self, eyecenter):
        # print('translate func {}'.format(eyecenter))
        if eyecenter:
            EyeCenter = Point(*eyecenter)
            x = self.screen_size.width * (EyeCenter.x - self.xeyeleft) / (self.xyeyright - self.xeyeleft)
            y = self.screen_size.height * (EyeCenter.y - self.yeyetop) / (self.yeyebottom - self.yeyetop)
            return (x, y)


if __name__ == '__main__':
    c = Calibrator((100, 100), calibration_data)
    print(c.translate((0, 0)))

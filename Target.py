import functools
import time

import numpy as np
import pygame

from Object import CoreObject, Speed

LOCATIONS = np.array([[1 / 2, 1 / 2], [0, -1], [-1, 1], [0, -1], [1 / 2, 1 / 2]])

def logging(func):
    begin = time.time()
    @functools.wraps(func)
    def inner(*args,**kwargs):
        print(time.time()-begin,args[0].center)
        func(*args,**kwargs)
    return inner

class Target(CoreObject):
    def __init__(self, screen, x, y, r, color=pygame.Color('red'), speed=(10, 10), calibration=False):
        CoreObject.__init__(self, x - r, y - r, 2 * r, 2 * r, speed)
        self.on_screen = screen  # Is used just to know the canvas size
        self.color = color
        self.radius = r
        self.x = x
        self.y = y
        self.calibration = calibration
        self.prev = time.time()
        self.counter = 0
    @logging
    def draw(self, surface):
        pygame.draw.circle(surface,
                           self.color,
                           # (255, 0, 0),
                           self.center,
                           self.radius)

    def update(self):

        width, height = self.on_screen.get_size()
        if not self.calibration:
            dx,dy = self.speed
            if self.left < 0 or self.right > width:
                dx = -dx
            if self.top < 0 or self.bottom > height:
                dy = -dy
            self.speed = Speed(dx,dy)

        else:
            size = np.array([width, height])
            delay = 1
            current_time = time.time()
            if current_time - self.prev > delay:
                self.prev = current_time
                print(" Event occured at {} ".format(current_time))
                self.speed = Speed._make((np.multiply(size, LOCATIONS).astype(int))[self.counter, :].tolist())
                self.counter = (self.counter + 1) % LOCATIONS.shape[0]
            else:
                self.speed = Speed(0, 0)

        super().update()

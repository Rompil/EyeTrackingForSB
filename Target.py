import pygame
from Object import CoreObject, Speed
import functools
import time

def logging(func):
    begin = time.time()
    @functools.wraps(func)
    def inner(*args,**kwargs):
        print(time.time()-begin,args[0].center)
        func(*args,**kwargs)
    return inner

class Target(CoreObject):
    def __init__(self, screen, x, y, r, color=pygame.Color('red'), speed=(10, 10), random=False):
        CoreObject.__init__(self, x - r, y - r, 2 * r, 2 * r, speed)
        self.on_screen = screen  # Is used just to know the canvas size
        self.color = color
        self.radius = r
        self.x = x
        self.y = y
        self.random = random
    @logging
    def draw(self, surface):
        pygame.draw.circle(surface,
                           self.color,
                           # (255, 0, 0),
                           self.center,
                           self.radius)

    def update(self):

        width, height = self.on_screen.get_size()
        if not self.random:
            dx,dy = self.speed

            if self.left < 0 or self.right > width:
                dx = -dx

            if self.top < 0 or self.bottom > height:
                dy = -dy

            self.speed = Speed(dx,dy)

            super().update()




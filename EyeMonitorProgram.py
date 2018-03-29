import pygame
# import sys

from Core import Core
from Target import Target
from CameraRecoder import CamRecoder


class MainLoop(Core):
    def __init__(self):
        super().__init__("Main window", 640,480, 60, fullscreen=False)

        def esc_func(dummy_param):
            "This func is needed it exit frm the program by ESC key"
            self.isFinished = True

        self.keydown_handlers[pygame.K_ESCAPE].append(esc_func)
        self.createObjects()

    def createObjects(self):
        "Add new objects here"
        self.create_Target()
        self.create_CamRecoder()

    def create_Target(self):
        self.objects.append(Target(self.screen, self.size.width // 2, self.size.height // 2, 10, speed=(5, 5)))

    def create_CamRecoder(self):
        self.objects.append((CamRecoder(self.screen)))
        self.objects[-1].start_record()

    def update(self):
        self.screen.fill(pygame.Color('gray17'))
        super().update()


def main():
    MainLoop().run()


if __name__ == '__main__':
    main()

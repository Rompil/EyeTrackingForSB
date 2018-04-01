import sys
from collections import defaultdict, namedtuple

import pygame


class Core:
    ''' This is a base class for our program.
    It implements basic features.
    '''

    def __init__(self, caption, width, height, frame_rate, fullscreen=True):

        self.frame_rate = frame_rate
        self.isFinished = False

        self.objects = []

        pygame.init()
        pygame.font.init()

        # Window name
        pygame.display.set_caption(caption)

        # Hide mouse cursor
        pygame.mouse.set_visible(False)

        # Store screen and size
        self.size = namedtuple('Screen_size', 'width height')(width, height)
        if fullscreen:
            self.screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((width, height))
            # windowed mode is for debug so we also show the cursor
            pygame.mouse.set_visible(True)

        self.clock = pygame.time.Clock()
        self.keydown_handlers = defaultdict(list)
        self.keyup_handlers = defaultdict(list)
        self.mouse_handlers = []

    def update(self):
        for o in self.objects:
            o.update()

    def draw(self):
        for o in self.objects:
            o.draw(self.screen)

    def finish(self):
        for o in self.objects:
            o.finish()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                for handler in self.keydown_handlers[event.key]:
                    handler(event.key)
            elif event.type == pygame.KEYUP:
                for handler in self.keydown_handlers[event.key]:
                    handler(event.key)
            elif event.type in (pygame.MOUSEBUTTONDOWN,
                                pygame.MOUSEBUTTONUP,
                                pygame.MOUSEMOTION):
                for handler in self.mouse_handlers:
                    handler(event.type, event.pos)

    def run(self):
        while not self.isFinished:
            self.handle_events()
            self.update()
            self.draw()

            pygame.display.update()
            self.clock.tick(self.frame_rate)
        self.finish()
        pygame.quit()

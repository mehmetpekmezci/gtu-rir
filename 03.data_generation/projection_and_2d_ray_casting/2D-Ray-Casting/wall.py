import pygame

# define wall class
class Wall:

    # constructor for Wall object
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # draws Wall
    def show(self, surface):
        pygame.draw.line(surface, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 5)
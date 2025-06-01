import pygame
import math
from ray import Ray

# define light class
class Light:

    # constructor for new light object
    def __init__(self, x1, y1, n):
        self.x1 = x1
        self.y1 = y1
        self.rays = []
        self.n = n
        for i in range(0, 360, int(360 / n)):
            self.rays.append(Ray(self.x1, self.y1, math.cos(math.radians(i)), math.sin(math.radians(i))))

    # draws light by checking for ray collisions and drawing rays appropriately
    def show(self, surface, walls):
        for ray in self.rays:
            ray.x1 = self.x1
            ray.y1 = self.y1
            closest = 1000000
            closestPoint = None
            for wall in walls:
                intersection = ray.collide(wall)
                if(intersection != False):
                    distance = math.sqrt((ray.x1 - intersection[0])**2 + (ray.y1 - intersection[1])**2)
                    if(distance < closest):
                        closest = distance
                        closestPoint = intersection

            if closestPoint is not None:
                pygame.draw.line(surface, (255,255,255), (ray.x1, ray.y1), closestPoint)
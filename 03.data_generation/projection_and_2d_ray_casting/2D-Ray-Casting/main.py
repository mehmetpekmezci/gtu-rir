import pygame
import math
import random

from pygame.locals import *
from light import Light
from wall import Wall
from ray import Ray


# initialize pygame
pygame.init()

# setup pygame display
screen_info = pygame.display.Info()
logo = pygame.image.load('images/logo.jpg')
pygame.display.set_icon(logo)
height, width = int(screen_info.current_h * 0.9), int(screen_info.current_w * 0.9)
display = pygame.display.set_mode((width, height))
pygame.display.set_caption('2DRayCasting')
display.fill((0,0,0))

# initialize constants
FPS = 30
CLOCK = pygame.time.Clock()
NUM_RAYS = 360 # defines number of rays being shown

# initialize border walls
walls = []
walls.append(Wall(0, 0, width - 1, 0)) # top Wall
walls.append(Wall(0, 0, 0, height - 1)) # left Wall
walls.append(Wall(0, height - 1, width - 1, height - 1)) # bottom Wall
walls.append(Wall(width - 1, 0, width - 1, height - 1)) # right Wall

# test Wall
w = Wall(500, 50, 500, 500)
# walls.append(w) * uncomment this line to activate test wall*

# initialize random non-border walls (currently randomizes between 0 and 10 walls, but can be modified as needed)
for i in range(random.randint(0, 10)):
    walls.append(Wall(random.randint(0, width), random.randint(0,height), random.randint(0, width), random.randint(0,height)))

# initialize light and its associated rays
l = Light(500, 500, NUM_RAYS)

# main loop
running = True

while running:
    # fill screen with black
    display.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:   
            running = False
        elif event.type == pygame.KEYDOWN:
            # regenerates new walls
            if event.key == K_r:
                # clear non-border walls
                del walls[4:len(walls)]
                # regenerate new non-border walls
                for i in range(random.randint(0, 10)):
                    walls.append(Wall(random.randint(0, width), random.randint(0,height), random.randint(0, width), random.randint(0,height)))

    # set x & y position of light to the current mouse position
    l.x1, l.y1 = pygame.mouse.get_pos()

    # draw Walls
    for wall in walls:
        wall.show(display)

    # draw light and its associated rays
    l.show(display, walls)

    #update display
    pygame.display.update()
    CLOCK.tick(FPS)

pygame.quit()
quit()
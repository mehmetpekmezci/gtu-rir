import pygame

# define ray class
class Ray:

    # constructor for new Ray object
    def __init__(self, x1, y1, dirX, dirY):
        self.x1 = x1
        self.y1 = y1
        self.dirX = dirX
        self.dirY = dirY

    # function to check if and where a ray collides with a wall object
    def collide(self, wall):
        # based off line segment intersection formula found at https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        
        # set variables associated with formula
        wx1 = wall.x1
        wy1 = wall.y1
        wx2 = wall.x2
        wy2 = wall.y2

        rx3 = self.x1
        ry3 = self.y1
        rx4 = self.x1 + self.dirX
        ry4 = self.y1 + self.dirY


        # numerator and denominator for the line-line intersection formula that calculates intersection of wall and ray.
        n = (wx1 - rx3) * (ry3 - ry4) - (wy1 - ry3) * (rx3 - rx4)
        d = (wx1 - wx2) * (ry3 - ry4) - (wy1 - wy2) * (rx3 - rx4)

        # check for invalid denominator
        if d == 0:
            return False

        t = n / d
        u = ((wx2 - wx1) * (wy1 - ry3) - (wy2 - wy1) * (wx1 - rx3)) / d

        # determines if there is a collision and returns intersection point as tuple
        if (t > 0 and t < 1 and u > 0):
            # calculate and return point of intersection
            px = (wx1 + t * (wx2 - wx1))
            py = (wy1 + t * (wy2 - wy1))
            return (px, py)
        else:
            return False
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # ignore numpy runtime warning saying divide by zero or nan

"""

Three main functions for raycasting:
raycast_ray_segment : one ray, one segment 
raycast_ray_segments : one ray, multi segments 
raycast_rays_segments : multi rays, multi segments 
  
ray: numpy array [p1, p2] (vector in space)
segment: numpy array [p1, p2] (segment in space)
return: the point of intersection alpha distance metric [x, y, a]
none if no intersection
"""
def raycast_ray_segment(ray, segment):
    r_px = ray[0, 0]
    r_py = ray[0, 1]
    r_dx = ray[1, 0] - ray[0, 0]
    r_dy = ray[1, 1] - ray[0, 1]

    s_px = segment[0, 0]
    s_py = segment[0, 1]
    s_dx = segment[1, 0] - segment[0, 0]
    s_dy = segment[1, 1] - segment[0, 1]

    r_mag = np.sqrt(r_dx * r_dx + r_dy * r_dy)
    s_mag = np.sqrt(s_dx * s_dx + s_dy * s_dy)

    if r_dx / r_mag == s_dx / s_mag and r_dy / r_mag == s_dy / s_mag:
        return None

    try:
        T2 = (r_dx * (s_py - r_py) + r_dy * (r_px - s_px)) / (s_dx * r_dy - s_dy * r_dx)
    except ZeroDivisionError:
        T2 = (r_dx * (s_py - r_py) + r_dy * (r_px - s_px)) / (s_dx * r_dy - s_dy * r_dx - 0.01)

    try:
        T1 = (s_px + s_dx * T2 - r_px) / r_dx
    except ZeroDivisionError:
        T1 = (s_px + s_dx * T2 - r_px) / (r_dx - 0.01)

    if T1 < 0: return None
    if T2 < 0 or T2 > 1: return None

    return (r_px + r_dx * T1, r_py + r_dy * T1, T1)

"""
ray: numpy array [p1, p2] (vector in space)
segments: numpy array [:, (p1, p2)] (segments in space)
return: a list of intersections and alpha distance metric [number_of_intersections, (x, y, a)]
none if no intersections
"""
def raycast_ray_segments(ray, segments):
    r_px = ray[0, 0]
    r_py = ray[0, 1]
    r_dx = ray[1, 0] - ray[0, 0]
    r_dy = ray[1, 1] - ray[0, 1]

    s_px = segments[:, 0, 0]
    s_py = segments[:, 0, 1]
    s_dx = segments[:, 1, 0] - segments[:, 0, 0]
    s_dy = segments[:, 1, 1] - segments[:, 0, 1]

    T2 = (r_dx * (s_py - r_py) + r_dy * (-s_px + r_px)) / (s_dx * r_dy - s_dy * r_dx)
    T1 = (s_px + s_dx * T2 - r_px) / r_dx

    # remove bad values
    T1 = T1[np.logical_and(np.invert(T1 < 0), np.invert(np.logical_or(T2 < 0, T2 > 1)))]
    return np.vstack((r_px + r_dx * T1, r_py + r_dy * T1, T1))

"""
rays: numpy array [:, p1, p2] (vectors in space)
segments: numpy array [:, (p1, p2)] (segments in space)
return: a list of intersections and alpha distance metric [rays, segments, (x, y, a)]
fully nan array if no intersections 

for example:
[0,0,:] would return the intersection of the first ray on the first segment, nan if no intersection 
[0,1,:] would return the intersection of the first ray on the first segment, nan if no intersection
[0,:,:] would return all the intersections of the first ray with each segment, very similar to the ray_segments function
nan values will be used for each segment that is not intersected ((x, y, a) => (nan,nan,nan))

The returned array is non-trivial to parse out all the data (takes some numpy functions)
The closest_intersection_from_raycast_rays_segments will return the closest hit for each ray
that has at least one intersection with a segment 
Not he alpha distance metric is smaller for closer intersections 
"""
def raycast_rays_segments(rays, segments):
    n_r = rays.shape[0]
    n_s = segments.shape[0]

    r_px = rays[:, 0, 0]
    r_py = rays[:, 0, 1]
    r_dx = rays[:, 1, 0] - rays[:, 0, 0]
    r_dy = rays[:, 1, 1] - rays[:, 0, 1]

    s_px = np.tile(segments[:, 0, 0], (n_r, 1))
    s_py = np.tile(segments[:, 0, 1], (n_r, 1))
    s_dx = np.tile(segments[:, 1, 0] - segments[:, 0, 0], (n_r, 1))
    s_dy = np.tile(segments[:, 1, 1] - segments[:, 0, 1], (n_r, 1))

    t1 = (s_py.T - r_py).T
    t2 = (-s_px.T + r_px).T
    t3 = (s_dx.T * r_dy).T
    t4 = (s_dy.T * r_dx).T
    t5 = (r_dx * t1.T).T
    t6 = (r_dy * t2.T).T
    t7 = t3 - t4
    t8 = (r_dx * t1.T).T
    t9 = (r_dy * t2.T).T

    T2 = (t8 + t9) / (t3 - t4)
    T1 = (((s_px + (s_dx * T2)).T - r_px) / r_dx).T

    ix = ((r_px + r_dx * T1.T).T)
    iy = ((r_py + r_dy * T1.T).T)
    
    ix=ix-np.sign(ix-r_px.reshape((r_px.shape[0],1)))*0.001*ix
    iy=iy-np.sign(iy-r_py.reshape((r_py.shape[0],1)))*0.001*iy
    
    intersections = np.stack((ix, iy, T1), axis=-1)

    bad_values = np.logical_or((T1 < 0), np.logical_or(T2 < 0, T2 > 1))
    intersections[bad_values, :] = np.nan

    return intersections

def generate_rays_from_points(view_position, points):
    angles = np.arctan2(points[:, 1] - view_position[1], points[:, 0] - view_position[0])
    # sort angles for correct polygon recontruction
    angles = np.flip(np.sort(np.concatenate((angles - 0.00001, angles + 0.00001))), 0)

    """ return unit vectors pointing to each point in the scene ...
        once the amount of points becomes very high, will become more
        efficent to just create equally spaced rays around the look position """
    rays = np.empty((angles.shape[0], 2, 2))
    rays[:, 0, 0] = view_position[0]
    rays[:, 0, 1] = view_position[1]
    rays[:, 1, 0] = view_position[0] + np.cos(angles)
    rays[:, 1, 1] = view_position[1] + np.sin(angles)

    return rays



def unique_points_from_segments(segments):
    all_points = segments.reshape(-1, 2)
    return np.unique(all_points, axis=0)

def closest_intersection_from_raycast_rays_segments(intersections):
    # get closest intersections (rays, segments, (x,y,T1))

    # remove rays with no intersection (full nan return on the final axis, causes nanargmin to throw error)
    # kinda obscure code
    n = (~np.isnan(intersections).any(axis=-1)).any(axis=-1)
    intersections = intersections[n,:,:]

    closest = np.nanargmin(intersections[:, :, 2], axis=1)
    return intersections[list(range(0, intersections.shape[0])), closest, :2]

def segments_from_box(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    s = []
    s.append([[x1,y1],[x2,y1]])
    s.append([[x2,y1],[x2,y2]])
    s.append([[x2,y2],[x1,y2]])
    s.append([[x1,y2],[x1,y1]])
    return s

def draw_segments(screen, segments):
    for p1, p2 in segments:
        pygame.draw.line(screen, (0,0,0), p1, p2, 1)

def generate_random_box_segments(w, h, number=1, size_min=20, size_max=100):
    x1 = np.random.randint(w, size=number) # random widths
    y1 = np.random.randint(h, size=number) # random heights
    x2 = x1 + np.random.randint(size_max-size_min, size=number)+size_min
    y2 = y1 + np.random.randint(size_max-size_min, size=number)+size_min

    p1 = np.column_stack((x1, y1))
    p2 = np.column_stack((x2, y2))
    p3 = np.column_stack((x2, y1))
    p4 = np.column_stack((x1, y2))

    s1 = np.column_stack((p1, p3)).reshape(-1,2,2)
    s2 = np.column_stack((p3, p2)).reshape(-1,2,2)
    s3 = np.column_stack((p2, p4)).reshape(-1,2,2)
    s4 = np.column_stack((p4, p1)).reshape(-1,2,2)

    return np.concatenate((s1, s2, s3, s4), axis=0)



if __name__ == "__main__":



    if True:
        import pygame
        from pygame.locals import *
        import sys

        pygame.init()
        width, height = 256,256
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Visual Demo")
        clock = pygame.time.Clock()

        segments = []
        segments.extend(segments_from_box((10,10), (width-10, height-10)))
        segments.extend(generate_random_box_segments(width, height, 50).tolist())
        segments = np.array(segments)

        points = unique_points_from_segments(segments)

        mouse_position = (0,0)
        method = True
        while True:
            #clock.tick(60)
            clock.tick(1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEMOTION:
                    mouse_position = event.pos
                if event.type == KEYDOWN:
                    if event.key == K_m:
                        method = not method

            screen.fill((255,255,255))

            print(mouse_position)
            
            rays = generate_rays_from_points(mouse_position, points)

            intersections = raycast_rays_segments(rays, segments)
            closest = closest_intersection_from_raycast_rays_segments(intersections)
#            pygame.draw.polygon(screen, (255,0,0), closest)
#            for intersect in closest:
#                pygame.draw.aaline(screen, (0, 255, 0), mouse_position, (intersect[0], intersect[1]))

            for origin in closest:
                    rays2=generate_rays_from_points(origin, points)
                    intersections2 = raycast_rays_segments(rays2, segments)
                    closest2=closest_intersection_from_raycast_rays_segments(intersections2)
                    pygame.draw.polygon(screen, (0,0,255), closest2)
                    for intersect2 in closest2:
                          pygame.draw.aaline(screen, (0, 255, 255), origin, (intersect2[0], intersect2[1]))

            pygame.draw.polygon(screen, (255,0,0), closest)
            for intersect in closest:
                pygame.draw.aaline(screen, (0, 255, 0), mouse_position, (intersect[0], intersect[1]))


            draw_segments(screen, segments)
            pygame.display.flip()
            pygame.display.set_caption("fps: " + str(clock.get_fps()))

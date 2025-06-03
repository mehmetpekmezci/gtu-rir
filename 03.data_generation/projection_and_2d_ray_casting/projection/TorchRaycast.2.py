import numpy as np
np.seterr(divide='ignore', invalid='ignore') # ignore numpy runtime warning saying divide by zero or nan
import pygame
from pygame.locals import *
import sys
import trimesh
import torch 
import time 
import threading
 
RESOLUTION=512
 
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
def raycast_rays_segments_(rays, segments):
    rays=torch.from_numpy(rays).cuda()
    segments=torch.from_numpy(segments).cuda()
    
    n_r = rays.shape[0]
    n_s = segments.shape[0]

    r_px = rays[:, 0, 0]
    r_py = rays[:, 0, 1]
    r_dx = rays[:, 1, 0] - rays[:, 0, 0]
    r_dy = rays[:, 1, 1] - rays[:, 0, 1]

    s_px = torch.tile(segments[:, 0, 0], (n_r, 1))
    s_py = torch.tile(segments[:, 0, 1], (n_r, 1))
    s_dx = torch.tile(segments[:, 1, 0] - segments[:, 0, 0], (n_r, 1))
    s_dy = torch.tile(segments[:, 1, 1] - segments[:, 0, 1], (n_r, 1))

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
    
    ix=ix-torch.sign(ix-r_px.reshape((r_px.shape[0],1)))*0.001*ix
    iy=iy-torch.sign(iy-r_py.reshape((r_py.shape[0],1)))*0.001*iy
    
    intersections = torch.stack((ix, iy, T1), axis=-1)

    bad_values = torch.logical_or((T1 < 0), torch.logical_or(T2 < 0, T2 > 1))
    intersections[bad_values, :] = torch.nan

    intersections=intersections.detach().cpu().numpy()
    
    return intersections

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


def segments_from_path2d(path2d,WIDTH,DEPTH):
        segments=[]
       
        #print(vertices)
        vertices=np.array(path2d.vertices)*MESH_EXPAND_RATIO
        vertices[:,1]=vertices[:,1]*-1
        vertices[:,0]+=(WIDTH/2)
        vertices[:,1]+=(DEPTH/2)

        for line in path2d.entities:
                line_segments=vertices[line.points]
                print(f"line_segments={line_segments}")
                if line_segments.shape[0]>2:
                     for i in range(1,line_segments.shape[0]-1):
                              segments.append(line_segments[i-1:i+1])
                else:
                     segments.append(line_segments)
        return segments

def draw_segments(screen, segments):
    for p1, p2 in segments:
        pygame.draw.line(screen, (0,0,0), p1, p2, 1)


def a(origin,points,segments):
                    rays2=generate_rays_from_points(origin, points)
                    intersections2 = raycast_rays_segments(rays2, segments)
                    closest2=closest_intersection_from_raycast_rays_segments(intersections2)
                    pygame.draw.polygon(screen, (0,0,255), closest2)
                    for intersect2 in closest2:
                          pygame.draw.aaline(screen, (0, 255, 255), origin, (intersect2[0], intersect2[1]))


if __name__ == "__main__":



    if True:


        full_graph_path=str(sys.argv[1]).strip()
        mesh = trimesh.load_mesh(full_graph_path)
        path3d= mesh.section(plane_origin=[0,0,0], plane_normal=[0, 1, 0]) 
        path2d,matrix_to_3D = path3d.to_2D()
        v=np.array(mesh.vertices)
        max_x=np.max(v[:,0])
        min_x=np.min(v[:,0])
        max_y=np.max(v[:,1])
        min_y=np.min(v[:,1])
        max_z=np.max(v[:,2])
        min_z=np.min(v[:,2])
        
        DEPTH=(max_x-min_x)
        WIDTH=(max_z-min_z)
        HEIGHT=(max_y-min_y)
        max_dim=max(DEPTH,WIDTH)
        MESH_EXPAND_RATIO=RESOLUTION/max_dim
        
        DEPTH=DEPTH*MESH_EXPAND_RATIO
        WIDTH=WIDTH*MESH_EXPAND_RATIO

        
        pygame.init()
        width, height = WIDTH*1.2,DEPTH*1.2
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Visual Demo")
        clock = pygame.time.Clock()

        segments=segments_from_path2d(path2d,WIDTH,DEPTH)
        segments = np.array(segments)
        segments=segments[:100]
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

            t1=time.time()
            screen.fill((255,255,255))

            t2=time.time()
            
            
            rays = generate_rays_from_points(mouse_position, points)
            
            t3=time.time()
            
            intersections = raycast_rays_segments(rays, segments)

            t4=time.time()

            closest = closest_intersection_from_raycast_rays_segments(intersections)
#            pygame.draw.polygon(screen, (255,0,0), closest)
#            for intersect in closest:
#                pygame.draw.aaline(screen, (0, 255, 0), mouse_position, (intersect[0], intersect[1]))

            t5=time.time()

            threads=[]
            for origin in closest:                    
                    t51=time.time()
                    t=threading.Thread(target=a,args=(origin,points,segments))
                    t.start()
                    threads.append(t)
                    t52=time.time()

            for t in threads:
                    t.join()
                    
            t6=time.time()

            pygame.draw.polygon(screen, (255,0,0), closest)
            for intersect in closest:
                pygame.draw.aaline(screen, (0, 255, 0), mouse_position, (intersect[0], intersect[1]))

            t7=time.time()

            draw_segments(screen, segments)
            pygame.display.flip()
            
            pygame.image.save( screen, 'screen.png' )
            
            print(f"t2-t1={t2-t1}")
            print(f"t3-t2={t3-t2}")
            print(f"t4-t3={t4-t3}")
            print(f"t5-t4={t5-t4}")
            print(f"t52-t51={t52-t51}")
            print(f"t6-t5={t6-t5}")
            print(f"t7-t6={t7-t6}")
            #pygame.display.set_caption("fps: " + str(clock.get_fps()))

import math
import logging
import csv
import glob
import sys
import os
import argparse
import numpy as np
import time
import random
import pickle
import matplotlib
import pymeshlab as ml
import PIL
import traceback
import trimesh
import copy
import pyglet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
from numba import jit
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

from src.colormap import generate_colormap
from src.mesh import load_mesh
from src.plot import plot_rays_and_mesh,plot_mesh,plot_points,plot_rays
from src.ray_face_intersection import rays_triangles_intersection
from src.ray_generator  import get_ray_directions,generate_ray_centers

np.set_printoptions(threshold=sys.maxsize)


PARAM_RESOLUTION_QUOTIENT=90 ## 180'i tam bolmesi lazim.
PARAM_NUMBER_OF_RAY_CENTERS=3
PARAM_MAX_FACES=1000 #2500


t0=time.time()
STATIC_RAY_DIRECTIONS=get_ray_directions(RESOLUTION_QUOTIENT=PARAM_RESOLUTION_QUOTIENT)
plot_rays(STATIC_RAY_DIRECTIONS)
t1=time.time()
#mesh=load_mesh("./data/a90118af-0020-44f0-8699-be48e0bdcbb6.obj", TOTAL_NUMBER_OF_FACES=PARAM_MAX_FACES,FORCE_DECIMATION=True)
#mesh=load_mesh("./data/gtu-cs-room-208.mesh.0.obj", TOTAL_NUMBER_OF_FACES=PARAM_MAX_FACES,FORCE_DECIMATION=True)
mesh=load_mesh("./data/floor_plan-1.mesh.obj", TOTAL_NUMBER_OF_FACES=PARAM_MAX_FACES,FORCE_DECIMATION=True)
#mesh=load_mesh("./data/room-z23-freecad-mesh-Body.obj", TOTAL_NUMBER_OF_FACES=PARAM_MAX_FACES,FORCE_DECIMATION=True)



plot_mesh(mesh,"./data/","a90118af-0020-44f0-8699-be48e0bdcbb6")

#sys.exit(0)
#plot_points(mesh.vertices,"./data/","a90118af-0020-44f0-8699-be48e0bdcbb6")
t2=time.time()
randomRayCenterPositions=generate_ray_centers(mesh,NUMBER_OF_RAY_CENTERS=PARAM_NUMBER_OF_RAY_CENTERS)
#print(f"{MAX_X}, { MAX_Y}, {MAX_Z}  === {np.max(randomRayCenterPositions[:,0])},{np.max(randomRayCenterPositions[:,1])},{np.max(randomRayCenterPositions[:,2])}")
t3=time.time()
tirangle_coordinates = mesh.vertices [mesh.faces .flatten()].reshape(-1,3,3) # for each face,  [ v1 [x, y, z] , v2 [x, y, z], v3 [x, y, z] ] 
#colormap=generate_colormap(STATIC_RAY_DIRECTIONS.shape[0])
t4=time.time()
for i in range(randomRayCenterPositions.shape[0]):
       all_rays_intersected, all_rays_t=rays_triangles_intersection(randomRayCenterPositions[i],STATIC_RAY_DIRECTIONS,tirangle_coordinates)
       
       indices=~np.isnan(all_rays_t)

       all_rays_t_min=np.min(np.nan_to_num(all_rays_t,nan=np.inf), axis=1)
       all_rays_t_argmin=np.argmin(np.nan_to_num(all_rays_t,nan=np.inf), axis=1).astype(int)
       all_rays_intersected_true_indices=np.argwhere(all_rays_intersected==True) 
       
       print(f"all_rays_intersected_true_indices.shape={all_rays_intersected_true_indices.shape}")
       print(f"all_rays_t_argmin.shape={all_rays_t_argmin.shape}")
       print(f"randomRayCenterPositions[i]={randomRayCenterPositions[i]}")
       print(f"all_rays_t_argmin={all_rays_t_argmin}")
       print(f"all_rays_t_min={all_rays_t_min}")
       print(mesh.faces[all_rays_t_argmin])
       print(mesh.vertices[mesh.faces[all_rays_t_argmin].flatten()])
       mesh2=trimesh.Trimesh(mesh.vertices,mesh.faces[all_rays_intersected_true_indices.flatten()])
       #mesh2=trimesh.Trimesh(mesh.vertices,mesh.faces[all_rays_t_argmin])
       print(f"mesh.faces.shape={mesh.faces.shape} mesh2.faces.shape={mesh2.faces.shape} all_rays_t.shape={all_rays_t.shape}")
       
       
       
       
       #colors=['#404040CC']*all_rays_t.shape[1]
       #print(all_rays_t_argmin)
       #print(all_rays_t_argmin.shape)
       
       #colors=np.array(colors)
       #colors[all_rays_t_argmin]=['#FF000000']
       
       
       ###mesh.faces=mesh.faces[all_rays_t_argmin]
       ###mesh=trimesh.Trimesh(mesh.faces,mesh.vertices)
       
       plot_rays_and_mesh('_'+str(i)+'_',randomRayCenterPositions[i],STATIC_RAY_DIRECTIONS,mesh,mesh2)#colors
       #print(f"all_rays_intersected.shape={all_rays_intersected.shape} all_rays_t_min.shape={all_rays_t_min.shape}")
       #print(all_rays_intersected)

        
       #plot_rays(STATIC_RAY_DIRECTIONS)

t5=time.time()



print(t1-t0)
print(t2-t1)
print(t3-t2)
print(t4-t3)
print(t5-t4)


      





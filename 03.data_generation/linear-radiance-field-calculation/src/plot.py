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

def plot_rays_and_mesh(file_name,ray_origin,ray_directions,mesh,mesh2):#,colors
   if len(mesh.faces)==0:
        return
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   #ax.set_fc(colors)
   #meshplot=ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1],  Z=mesh.vertices[:,2],triangles=mesh.faces,alpha = 0.1,cmap=matplotlib.cm.Blues)
   #ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1],mesh.vertices[:,2],triangles=mesh.faces,alpha = 0.2,cmap=ListedColormap(colors))
   ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1],mesh.vertices[:,2],triangles=mesh.faces,alpha = 0.2,color='k')
   ax.plot_trisurf(mesh2.vertices[:, 0], mesh2.vertices[:,1],mesh2.vertices[:,2],triangles=mesh2.faces,alpha = 0.9,color='r')
   
   x=np.full(ray_directions.shape[0],ray_origin[0])
   y=np.full(ray_directions.shape[0],ray_origin[1])
   z=np.full(ray_directions.shape[0],ray_origin[2])
   ax.text(0, 0,3,f"origin={float(ray_origin[0]):.1f},{float(ray_origin[1]):.1f},{float(ray_origin[2]):.1f}\n", style='italic', bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10})
   #ax.text(-10, 0,4,f"origin={ray_directions}\n", style='italic', bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10})
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   # Make the direction data for the arrows
   u = ray_directions[:,0]
   v = ray_directions[:,1]
   w = ray_directions[:,2]
   ax.quiver(x, y, z, u, v, w, color=['r'],length=4,  arrow_length_ratio=0.000001,normalize=True) #, 

   ax.scatter(ray_origin[0],[ray_origin[1]],[ray_origin[2]],color='k')
   ax.scatter([ray_origin[0]],[ray_origin[1]],[0],color='k')
   ax.scatter([0],[ray_origin[1]],[ray_origin[2]],color='k')
   ax.scatter([ray_origin[0]],[0],[ray_origin[2]],color='k')
   #ax.plot(ray_origin,[0,ray_origin[1],ray_origin[2]],color='b')
   ax.plot([ray_origin[0], 0], [ray_origin[1],ray_origin[1]],zs=[ray_origin[2],ray_origin[2]],color='b')
   ax.plot([ray_origin[0], -10], [ray_origin[1],ray_origin[1]],zs=[ray_origin[2],ray_origin[2]],color='b')
   ax.plot([ray_origin[0], ray_origin[0]], [ray_origin[1],0],zs=[ray_origin[2],ray_origin[2]],color='b')
   ax.plot([ray_origin[0], ray_origin[0]], [ray_origin[1],-10],zs=[ray_origin[2],ray_origin[2]],color='b')
   ax.plot([ray_origin[0], ray_origin[0]], [ray_origin[1],ray_origin[1]],zs=[ray_origin[2],0],color='b')
   ax.plot([ray_origin[0], ray_origin[0]], [ray_origin[1],ray_origin[1]],zs=[ray_origin[2],-1],color='b')
#   x1, = [-1, 12], [1, 4]
#   x2,  = [1, 10], [3, 2]
#   plt.plot(x1, y1, marker = 'o')


   plt.savefig("./output/graph."+file_name+".rays_and_meshes.png")
   
#   ax.view_init(0, 90,0)
#   plt.savefig("./output/graph."+file_name+".rays_and_meshes.0.30.0.png")
   ax.view_init(elev=30, azim=45, roll=15)
   plt.savefig("./output/graph."+file_name+".rays_and_meshes.30.45.15.png")
   ax.view_init(elev=45, azim=120, roll=15)
   plt.savefig("./output/graph."+file_name+".rays_and_meshes.45.120.15.png")
#   ax.view_init(90,  0,0)
#   plt.savefig("./output/graph."+file_name+".rays_and_meshes.90.0.0.png")
#   ax.view_init(90, 0,0)
#   plt.savefig("./output/graph."+file_name+".rays_and_meshes.30.0.0.png")
  
   plt.close()
##    ax.scatter(mesh.vertices[:10, 2],mesh.vertices[:10,0],mesh.vertices[:10,1],color=['red','green','blue','orange','purple','brown','pink','gray','olive','cyan'],s=[20,40,60,80,100,120,140,160,180,200])

     
def plot_mesh(mesh,MESH_dir,file_name):
    if len(mesh.faces)==0:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1],  Z=mesh.vertices[:,2],triangles=mesh.faces,alpha = 0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("./output/graph."+file_name+".png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:, 0],mesh.vertices[:,1],mesh.vertices[:,2])
    plt.savefig("./output/graph."+file_name+"-scatter.png")
    plt.close()

#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(mesh.vertices[:10, 0],mesh.vertices[:10,1],mesh.vertices[:10,2],color=['red','green','blue','orange','purple','brown','pink','gray','olive','cyan'],s=[20,40,60,80,100,120,140,160,180,200])
#    plt.savefig("./output/graph."+file_name+"-scatter.10points.png")
#    plt.close()



def plot_points(points,MESH_dir,file_name):
    if len(points)==0:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2],points[:,0],points[:,1])
    plt.savefig("./output/graph."+file_name+"-scatter-direct-points.png")
    plt.close()


def plot_rays(ray_directions):
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     #x, y, z = np.meshgrid(np.arange(-1, 1, 0.2), np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.8))
     x=y=z=np.zeros(ray_directions.shape[0])
     # Make the direction data for the arrows
     u = ray_directions[:,0]
     v = ray_directions[:,1]
     w = ray_directions[:,2]
     ax.quiver(x, y, z, u, v, w, color=['r','b','g'],length=1,  arrow_length_ratio=0.000001,normalize=True) #, 
     plt.savefig("./output/graph.ray.directions.png")
     plt.close()
     
     

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


def load_mesh(path, TOTAL_NUMBER_OF_FACES=2000,FORCE_DECIMATION=True):
    pymeshlab_mesh = ml.MeshSet()
    try :
        pymeshlab_mesh.load_new_mesh(path)
        pmesh=pymeshlab_mesh.current_mesh()

#        pymeshlab_mesh.apply_filter('meshing_remove_unreferenced_vertices')
#        pymeshlab_mesh.apply_filter('meshing_remove_duplicate_faces')
#        pymeshlab_mesh.apply_filter('meshing_remove_null_faces')
#        pymeshlab_mesh.apply_filter('meshing_poly_to_tri')
#        pymeshlab_mesh.apply_filter('meshing_repair_non_manifold_edges')



#        pymeshlab_mesh.apply_filter('meshing_surface_subdivision_midpoint',iterations =6)
#        print(len(pmesh.face_matrix()))
#        pymeshlab_mesh.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=TOTAL_NUMBER_OF_FACES, preservenormal=True)
        mesh = trimesh.Trimesh(pmesh.vertex_matrix(),pmesh.face_matrix())
    except:
        print(f"{path} file is imported by pymeshlab but thrown an error")
        mesh = trimesh.load_mesh(path, process=False)

    ### hala buyukse mesh.faces, o zaman elle silecegim. (bu duruma dusmemesi lazim, gelistirmenin az GPUlu makinede devam edebilmesi icin yapiyorum)
    if mesh.faces.shape[0] > TOTAL_NUMBER_OF_FACES and FORCE_DECIMATION :
         mesh.faces=mesh.faces[:TOTAL_NUMBER_OF_FACES]


   # ORIENTATION
    V=np.empty(mesh.vertices.shape)
    
    V[:,0]=mesh.vertices[:,0]
    V[:,1]=-mesh.vertices[:,2]
    V[:,2]=mesh.vertices[:,1]
    ##GTURIR ORIENTATION
    if   "gtu-cs-room-"  in path or  "-freecad-mesh-Body" in path:
           V[:,1]=V[:,1]+np.max(mesh.vertices[:,2])
    
    mesh=trimesh.Trimesh(V,mesh.faces)
    return mesh
#    F = mesh.faces ## EVERY FACE IS COMPOSED OF 3 node NUMBERS (vertex index): 
#    V = mesh.vertices # this gives every verteice's x,y,z coordinates.
#    tirangle_coordinates = V[F.flatten()].reshape(-1,9) # for each face,  [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ] 
#    centers = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1) # for each face,  [ (v1.x+v2.x+v3.x)/3 , (v1.y+v2.y+v3.y)/3 ,(v1.z+v2.z+v3.z)/3 ]
#
#    normals = mesh.face_normals
#    areas = mesh.area_faces
#    ## GRAPH DECIMATION FACE sayisinin 1 veya 2 eksigini verebiliyor.
#    for i in range(TOTAL_NUMBER_OF_FACES-centers.shape[0]):
#       tirangle_coordinates=np.vstack((tirangle_coordinates,tirangle_coordinates[0]))
#       centers=np.vstack((centers,centers[0]))
#       normals=np.vstack((normals,normals[0]))
#       areas=areas.reshape(-1,1)
#       areas=np.vstack((areas,areas[0]))
#       areas=areas.reshape(-1)
#
#    tirangle_coordinates=np.array(tirangle_coordinates)
#    normals=np.array(normals)
#    centers=np.array(centers)
#    areas=np.array(areas).reshape(-1,1)
#    return tirangle_coordinates,normals,centers,areas


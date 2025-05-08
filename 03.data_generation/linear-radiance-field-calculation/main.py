import pymeshlab as ml
import time
import soundfile as sf
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from scipy import signal

import io
import sys
import random
import scipy.sparse as sp
import traceback


import json
import random
from pathlib import Path
import numpy as np
import os
import trimesh
from scipy.spatial.transform import Rotation
import copy
import csv

import math
import pyglet

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def load_mesh(path, TOTAL_NUMBER_OF_FACES=2000,FORCE_DECIMATION=True):
    pymeshlab_mesh = ml.MeshSet()
    try :
        pymeshlab_mesh.load_new_mesh(path)
        pmesh=pymeshlab_mesh.current_mesh()

        pymeshlab_mesh.apply_filter('meshing_remove_unreferenced_vertices')
        pymeshlab_mesh.apply_filter('meshing_remove_duplicate_faces')
        pymeshlab_mesh.apply_filter('meshing_remove_null_faces')
        pymeshlab_mesh.apply_filter('meshing_poly_to_tri')
        pymeshlab_mesh.apply_filter('meshing_repair_non_manifold_edges')
        #pymeshlab_mesh.apply_filter('meshing_repair_non_manifold_vertices')
        pymeshlab_mesh.apply_filter('meshing_surface_subdivision_midpoint')
        pymeshlab_mesh.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=TOTAL_NUMBER_OF_FACES, preservenormal=True)
        mesh = trimesh.Trimesh(pmesh.vertex_matrix(),pmesh.face_matrix())
    except:
        print(f"{path} file is imported by pymeshlab but thrown an error")
        mesh = trimesh.load_mesh(path, process=False)

    ### hala buyukse mesh.faces, o zaman elle silecegim. (bu duruma dusmemesi lazim, gelistirmenin az GPUlu makinede devam edebilmesi icin yapiyorum)
    if mesh.faces.shape[0] > TOTAL_NUMBER_OF_FACES and FORCE_DECIMATION :
         mesh.faces=mesh.faces[:TOTAL_NUMBER_OF_FACES]

    mesh=trimesh.Trimesh(mesh.vertices,mesh.faces)
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



def plot_mesh(mesh,MESH_dir,file_name):
    if len(mesh.faces)==0:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 2], mesh.vertices[:,0], triangles=mesh.faces, Z=mesh.vertices[:,1])
    plt.savefig(MESH_dir+"/graph."+file_name+".png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:, 2],mesh.vertices[:,0],mesh.vertices[:,1])
    plt.savefig(MESH_dir+"/graph."+file_name+"-scatter.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:10, 2],mesh.vertices[:10,0],mesh.vertices[:10,1],color=['red','green','blue','orange','purple','brown','pink','gray','olive','cyan'],s=[20,40,60,80,100,120,140,160,180,200])
    plt.savefig(MESH_dir+"/graph."+file_name+"-scatter.10points.png")
    plt.close()



def plot_points(points,MESH_dir,file_name):
    if len(points)==0:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2],points[:,0],points[:,1])
    plt.savefig(MESH_dir+"/graph."+file_name+"-scatter-direct-points.png")
    plt.close()



mesh=load_mesh("./data/a90118af-0020-44f0-8699-be48e0bdcbb6.obj", TOTAL_NUMBER_OF_FACES=2000,FORCE_DECIMATION=True)
plot_mesh(mesh,"./data/","a90118af-0020-44f0-8699-be48e0bdcbb6")
plot_points(mesh,"./data/","a90118af-0020-44f0-8699-be48e0bdcbb6")

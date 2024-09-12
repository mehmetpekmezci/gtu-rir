from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
# from PIL import Image
import soundfile as sf
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch_geometric
from torch_geometric.io import read_ply
import librosa

import io
import sys
import random
from miscc.config import cfg
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_adj
import scipy.sparse as sp
import traceback


from miscc.utils import save_mesh_as_obj,save_pos_face_as_obj,load_pickle, write_pickle


# using the modelnet40 as the dataset, and using the processed feature matrixes
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import pygem
from pygem import FFD
import copy
import csv

import math
import pyglet


#embeddings = [mesh_path,RIR_path,source,receiver]
class MeshDataset(data.Dataset):
    def __init__(self, data_dir,mesh_paths, train=True,augment=None): 
        self.data_dir = data_dir       
        self.mesh_paths = mesh_paths

        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
         
        if train and augment:
            self.augments = augment

    def __getitem__(self, index):

        label = 0
        pickle_file_path=os.path.join(self.data_dir,self.mesh_paths[index])
        mesh_pickle_file_path = pickle_file_path.replace('.pickle',f'.facecount-{cfg.MAX_FACE_COUNT}.pickle')
        full_mesh_path = pickle_file_path.replace('.pickle','.obj')
        pickle_file_content=None
        if(os.path.exists(mesh_pickle_file_path)):
            pickle_file_content=load_pickle(mesh_pickle_file_path)
        else:
            pickle_file_content=None

        if pickle_file_content is not None:
             (tirangle_coordinates,normals,centers,areas)=pickle_file_content
        else:
            tirangle_coordinates,normals,centers,areas = load_mesh(full_mesh_path, augments=self.augments,request=self.feats)
            tirangle_coordinates,normals,centers,areas = normalize_mesh_values(tirangle_coordinates,normals,centers,areas)
            #write_pickle(mesh_pickle_file_path,(tirangle_coordinates,normals,centers,areas))

        return   tirangle_coordinates, normals,centers,areas, full_mesh_path

        
    def __len__(self):
        return len(self.mesh_paths)
            
            
        


class TextDataset(data.Dataset):
    def __init__(self, data_dir,embeddings, split='train',rirsize=4096): #, transform=None, target_transform=None):

        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.bbox = None
        
  
        self.embeddings = embeddings

    def get_RIR(self, full_RIR_path):
        # wav,fs = sf.read(full_RIR_path) 
        wav,fs = librosa.load(full_RIR_path)
 
        # wav_resample = librosa.resample(wav,16000,fs)
        wav_resample = librosa.resample(wav,orig_sr=fs,target_sr=16000)

        length = wav_resample.size

        crop_length = 3968 #int(16384)
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            std_value = np.std(wav_resample) * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new = np.concatenate([wav_resample,zeros])/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])
        else:
            wav_resample_new = wav_resample[0:crop_length]
            std_value = np.std(wav_resample_new)  * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new =wav_resample_new/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])

        resample_length = int(self.rirsize)
        
        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        return RIR


    def get_graph(self, full_graph_path):
        
        full_mesh_path = full_graph_path.replace('.pickle','.obj')
        tirangle_coordinates,normals,centers,areas = load_mesh(full_mesh_path)
        
        graph={}
        
        tirangle_coordinates,normals,centers,areas = normalize_mesh_values(tirangle_coordinates,normals,centers,areas)
        graph={}
        graph["triangle_coordinates"]=tirangle_coordinates
        graph["normals"]=normals
        graph["centers"]=centers
        graph["areas"]=areas
   
        
        return graph
        
        
        #graph = load_pickle(full_graph_path)
        
        #return graph #edge_index, vertex_position


    def __getitem__(self, index):

        graph_path,RIR_path,source_location,receiver_location= self.embeddings[index]

        data_dir = self.data_dir

        full_graph_path = os.path.join(data_dir,graph_path)
        full_RIR_path  = os.path.join(data_dir,RIR_path)
        source_receiver = source_location+receiver_location

        RIR = self.get_RIR(full_RIR_path)

        graph = self.get_graph(full_graph_path)
        
        graph["RIR"] = RIR
        graph["embeddings"] =  np.array(source_receiver).astype('float32')

        return graph
        
    def __len__(self):
        return len(self.embeddings)








def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    vertices = mesh1.vertices - mesh1.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh1.vertices = vertices
    return mesh1


def mesh_deformation(mesh: trimesh.Trimesh):
    ffd = FFD([2, 2, 2])
    random = np.random.rand(6) * 0.1
    ffd.array_mu_x[1, 1, 1] = random[0]
    ffd.array_mu_y[1, 1, 1] = random[1]
    ffd.array_mu_z[1, 1, 1] = random[2]
    ffd.array_mu_x[0, 0, 0] = random[3]
    ffd.array_mu_y[0, 0, 0] = random[4]
    ffd.array_mu_z[0, 0, 0] = random[5]
    vertices = mesh.vertices
    new_vertices = ffd(vertices)
    mesh.vertices = new_vertices
    return mesh



def convert_normal_to_rotation_matrix(normal):
    #https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
    nx=normal[0]
    ny=normal[1]
    nz=normal[2]
    rotation= [  
                 [ ny/(math.sqrt(nx**2+ny**2)+0.0000001)     , -nx/(math.sqrt(nx**2+ny**2)+0.0000001)   ,   0                       ],
                 [ nx*nz/(math.sqrt(nx**2+ny**2)+0.0000001)  , ny*nz/(math.sqrt(nx**2+ny**2)+0.0000001) ,  -math.sqrt(nx**2+ny**2)  ],
                 [ nx                            , ny                           ,   nz                      ]
               ]
    return rotation      
                
def face_normal_center_area_to_pos_and_face(normals,centers,areas):
    pos=[]
    faces=[]
#    print(f"normals.shape={normals.shape} normals[:3]={normals[:3]}")
#    print(f"centers.shape={centers.shape} centers[:3]={centers[:3]}")
#    print(f"areas.shape={areas.shape} areas[:3]={areas[:3]}")

    #MP: ASSUME EQUILATERAL TRIANGLE
    # AREA= sqrt(3)/4 * a^2  ->  a= sqrt(4*AREA/sqrt(3))
    for index in range(len(areas)):
      
        area=areas[index]
        g=centers[index]
        n=normals[index]
      
        if(area>0):  
          a=math.sqrt(2*area)
        else:
          a=0
        ## MP : FIRST GROW A RIGHT TRIANGLE (ENLARGE IT TO HAVE THE CORRESPONDING area)
        posVertex1=[0  , 0  , 0 ]     # v1
        posVertex2=[a  , 0  , 0 ]     # v2
        posVertex3=[0  , a  , 0 ]     # v3
        
        ## MP: THEN RORATE IT PARRALEL TO NORMAL VECTOR
        rotation_matrix=np.array(convert_normal_to_rotation_matrix(n))
           
        triangle=[]
        triangle.append(posVertex1)
        triangle.append(posVertex2)
        triangle.append(posVertex3)
        triangle=np.array(triangle)
        triangle=triangle
       
        dot_product_with_normal=np.dot(triangle,n)
        all_zeros = not dot_product_with_normal.any()
        if not all_zeros:
           triangle=np.dot(triangle,rotation_matrix) #+ [[0.01,0,0],[0,0.01,0],[0,0,0.01]]       
           ## MP: if the triangle is perpendicular to normal vector, we should not apply rotation matrix.
    
        
        ## MP: THEN TRANSLATE IT TO THE CENTER
        translation_matrix=np.array( [ [g[0],g[1],g[2]],[g[0],g[1],g[2]],[g[0],g[1],g[2]] ])
        triangle=triangle+translation_matrix
        
 
        pos.append(triangle[0])
        pos.append(triangle[1])
        pos.append(triangle[2])
        

        faces.append([3*index,3*index+1,3*index+2])
        # a^2-a^2/4=9x^2,   , 3x is the height of the equilateral tringle, g point is at point x of equilateral triangle.
        #
        #          /|x\                v1
        #     a   / |x \              /  \
        #        /  |x  \           /  g  \
        #       ---------          v2 --- v3
        #        a/2  a/2
        # x= a/sqrt(12)
        #
        #        
        #                 x  
        #                 I
        #             y ---
        #                z=dışarı doğru
        
    return pos,faces
            
def save_face_normal_center_area_as_obj(normals,centers,areas,path):
    pos,faces=face_normal_center_area_to_pos_and_face(normals,centers,areas)
    save_pos_face_as_obj(pos,faces,path)




def normalize_mesh_values(triangle_coordinates,normals,centers,areas):

    # normalize all values between 0 and 1

    triangle_coordinates=(triangle_coordinates/cfg.MAX_DIM+1)/2
    normals=(normals+1)/2
    centers=(centers/cfg.MAX_DIM + 1 )/2
    areas=(areas/cfg.MAX_DIM+1)/2 

    return triangle_coordinates,normals,centers,areas

def denormalize_mesh_values(triangle_coordinates,normals,centers,areas):

    # normalize all values between 0 and 1
    
    triangle_coordinates=(triangle_coordinates*2-1)*cfg.MAX_DIM
    normals=normals*2-1
    centers=(centers*2-1)*cfg.MAX_DIM
    areas=(areas*2-1)*cfg.MAX_DIM

    return triangle_coordinates,normals,centers,areas


def load_mesh(path, augments=[], request=[], seed=None):
    TOTAL_NUMBER_OF_FACES=cfg.MAX_FACE_COUNT
    mesh = trimesh.load_mesh(path, process=False)
#    print('BEGINNING : mesh has ', mesh.vertices.shape[0], ' vertex and ', mesh.faces.shape[0], ' faces')
    if mesh.faces.shape[0] < TOTAL_NUMBER_OF_FACES-100 :
        while mesh.faces.shape[0] < TOTAL_NUMBER_OF_FACES:
              mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)


#    print(f"1.simplify_quadric_decimation TOTAL_NUMBER_OF_FACES={TOTAL_NUMBER_OF_FACES}")
#    while mesh.faces.shape[0] > TOTAL_NUMBER_OF_FACES :
    if mesh.faces.shape[0] > TOTAL_NUMBER_OF_FACES :
# MP BURADA DONUYOR DATALOADER
       mesh=mesh.simplify_quadric_decimation(TOTAL_NUMBER_OF_FACES)

#       print('AFTER SUBDIVIDE/DECIMATION : mesh has ', mesh.vertices.shape[0], ' vertex and ', mesh.faces.shape[0], ' faces')

#    print("2.simplify_quadric_decimation")

    
    ## hala buyukse mesh.faces, o zaman elle silecegim. (bu duruma dusmemesi lazim, gelistirmenin az GPUlu makinede devam edebilmesi icin yapiyorum)
    if mesh.faces.shape[0] > TOTAL_NUMBER_OF_FACES and cfg.FORCE_DECIMATION_EVEN_IF_THE_SHAPE_WILL_DEFORM_AND_USELESS :
         mesh.faces=mesh.faces[:TOTAL_NUMBER_OF_FACES]
           
       


       #mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],faces=[[0, 1, 2]])
                       
                       
                       

#    print('AFTER SUBDIVIDE/DECIMATION : mesh has ', mesh.vertices.shape[0], ' vertex and ', mesh.faces.shape[0], ' faces')

#    for method in augments:
#        if method == 'orient':
#            mesh = randomize_mesh_orientation(mesh)
#        if method == 'scale':
#            mesh = random_scale(mesh)
#        if method == 'deformation':
#            mesh = mesh_deformation(mesh)

    F = mesh.faces ## EVERY FACE IS COMPOSED OF 3 node NUMBERS (vertex index): 
    V = mesh.vertices # this gives every verteice's x,y,z coordinates.
    tirangle_coordinates = V[F.flatten()].reshape(-1,9) # for each face,  [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ] 
    centers = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1) # for each face,  [ (v1.x+v2.x+v3.x)/3 , (v1.y+v2.y+v3.y)/3 ,(v1.z+v2.z+v3.z)/3 ]

    normals = mesh.face_normals
    #print("=========")
    #print(f"centers = V[F.flatten()].reshape(-1, 3, 3)={V[F.flatten()].reshape(-1, 3, 3)}")
    #print(f"centers1 = V[F.flatten()].reshape(-1, 3, 3)={V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)}")
    #print(f"normals[:3]={normals[:3]}")
    ### MEHMET PEKMEZCI : CENTER'LAR TRIANGLE PLANE'inin ICINDE, ISPAT ETTIM.
    
    areas = mesh.area_faces
#    save_face_normal_center_area_as_obj(normals,centers,areas,path+".face.regenerated.1.obj")
#    save_as_obj(mesh.vertices,mesh.faces,path+"face.augmented.faces.1.obj")

    ## GRAPH DECIMATION FACE sayisinin 1 veya 2 eksigini verebiliyor.
    for i in range(TOTAL_NUMBER_OF_FACES-centers.shape[0]):
       tirangle_coordinates=np.vstack((tirangle_coordinates,tirangle_coordinates[0]))
       centers=np.vstack((centers,centers[0]))
       normals=np.vstack((normals,normals[0]))
       areas=areas.reshape(-1,1)
       areas=np.vstack((areas,areas[0]))
       areas=areas.reshape(-1)

    #tirangle_coordinates=torch.Tensor(np.array(tirangle_coordinates))
    tirangle_coordinates=np.array(tirangle_coordinates)
    #normals=torch.Tensor(np.array(normals))
    normals=np.array(normals)
    #centers=torch.Tensor(np.array(centers))
    centers=np.array(centers)
    areas=np.array(areas).reshape(-1,1)
    #areas=torch.Tensor(np.array(areas))
    return tirangle_coordinates,normals,centers,areas







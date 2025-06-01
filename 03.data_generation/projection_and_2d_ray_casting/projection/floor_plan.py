import time
import torch.utils.data as data
import os
import numpy as np
import torch
import librosa
import sys
from PIL import Image
import io
import pickle
import trimesh
import time


t=[]

t.append(time.time()) #t0

full_graph_path=str(sys.argv[1]).strip()

t.append(time.time()) #t1

mesh = trimesh.load_mesh(full_graph_path)

t.append(time.time()) #t2

RESOLUTION=256

t.append(time.time()) #t3

v=np.array(mesh.vertices)
max_x=np.max(v[:,0])
min_x=np.min(v[:,0])
max_y=np.max(v[:,1])
min_y=np.min(v[:,1])
max_z=np.max(v[:,2])
min_z=np.min(v[:,2])
room_dims=[(max_x-min_x)*100,(max_y-min_y)*100,(max_z-min_z)*100]

WIDTH=int(room_dims[2])
DEPTH=int(room_dims[0])

t.append(time.time()) #t4

max_dim=max(WIDTH,DEPTH)


scale=RESOLUTION/max_dim

WIDTH=int(WIDTH*scale)

DEPTH=int(DEPTH*scale)

t.append(time.time()) #t5

path3d= mesh.section(plane_origin=[0,0,0], plane_normal=[0, 1, 0]) 
t.append(time.time()) #t6
path2d,matrix_to_3D = path3d.to_2D()

t.append(time.time()) #t7

print(path2d.vertices)
for line in path2d.entities:
   print(line.points)
   print("----")



image=Image.open(io.BytesIO(path2d.scene().save_image(resolution=[WIDTH,DEPTH],visible=False)))

t.append(time.time()) #t8

image.save(full_graph_path+'.png')

t.append(time.time()) #t9

#for i in range(1,len(t)):
#    print(f"{i}-{i-1} : {t[i]-t[i-1]}")



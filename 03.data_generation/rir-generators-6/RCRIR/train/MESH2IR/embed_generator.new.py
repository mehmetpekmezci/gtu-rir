import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys

import time
import torch.utils.data as data
import torch
import librosa
import sys
import bpy
import bmesh
from miscc.config import cfg,cfg_from_file
from PIL import Image
from diffusers.models import AutoencoderKL
import threading
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)

cfg_from_file("cfg/RIR_s1.yml")

print(cfg)

rirsize = cfg.RIRSIZE
bpy_context = bpy.context
#bpy_context.scene.cycles.device = 'GPU'
bpy_scene = bpy_context.scene
bpy_depsgraph=bpy_context.evaluated_depsgraph_get()

image_vae = AutoencoderKL.from_pretrained("zelaki/eq-vae")
image_vae.eval()



#embeddings = [mesh_path,RIR_path,source,receiver]

training_embedding_list=[]
validation_embedding_list=[]

#path = "dataset/"
path = str(sys.argv[1]).strip()
folderstart = str(sys.argv[2]).strip()
CUDANO=2
if folderstart.isdigit() and int(folderstart)<8:
    CUDANO=1
CUDANO=str(CUDANO)    
if cfg.CUDA:
   image_vae.to(device='cuda:'+CUDANO)
mesh_folders = os.listdir(path)
# num_counter = 9
# temp_counter = 0 
# print("len folders ",len(mesh_folders))

def get_RIR(full_RIR_path):

        picklePath=full_RIR_path.replace(".wav",".pickle")

        if os.path.exists(picklePath):
            with open(picklePath, "rb") as f:
                  x = pickle.load(f)
            return x

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

        resample_length = int(rirsize)
        
        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        with open(picklePath, 'wb') as f:
            pickle.dump(RIR, f, protocol=2)

        return RIR

def clear_scene():
      for obj in bpy.context.scene.objects:
       if obj.type == 'MESH':
          obj.select_set(True)
       else:
          obj.select_set(False)
      bpy.ops.object.delete()
      for block in bpy.data.meshes:
         if block.users == 0:
             bpy.data.meshes.remove(block)
      for block in bpy.data.materials:
         if block.users == 0:
             bpy.data.materials.remove(block)
      for block in bpy.data.textures:
         if block.users == 0:
             bpy.data.textures.remove(block)
      for block in bpy.data.images:
         if block.users == 0:
             bpy.data.images.remove(block)

def ray_cast_per_alfa(ray_casting_image,origin,alfa,beta):
          direction=ray_directions[alfa][beta]
          hit, loc, normal, idx, obj, mw = bpy_scene.ray_cast(bpy_depsgraph,origin, direction)
          if hit:
              distance=np.linalg.norm(np.array(origin)-np.array(loc))
              gray_scale_color=min(int(63+192*distance/cfg.MAX_RAY_CASTING_DISTANCE),255)
              ray_casting_image[alfa][beta]=gray_scale_color
#### NOT : MP : CALISMAZSA BIR DE NORMALI DE ISIN ICINE KATMAYI DENEYEBILIRIZ              
              #print(f"HIT: location={np.array(loc).shape} normal_vector_of_hit_point={np.array(norm).shape} idx={np.array(idx).shape} obj={np.array(obj).shape} mw={np.array(mw).shape}")
          else:
              ray_casting_image[alfa][beta]=0

def ray_cast(bmesh_object,origin):
       origin=list(np.array(origin).astype(np.float32))
       ray_casting_image=np.zeros((len(ray_directions.keys()),len(ray_directions[0].keys())))
       for alfa in ray_directions:
        for beta in ray_directions[alfa]:
           ray_cast_per_alfa(ray_casting_image,origin,alfa,beta)
       ray_casting_image=ray_casting_image.reshape(cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION,1).repeat(3,axis=2)
       #ray_casting_image=Image.fromarray(np.uint8(ray_casting_image), mode="RGB")
       #ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32).reshape(1,3,cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION)
       ray_casting_image=torch.tensor(np.array(ray_casting_image.astype(np.uint8)).transpose(2, 0, 1), dtype=torch.float32).reshape(1,3,cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION)
       #ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32)

       return ray_casting_image


def generate_ray_directions():
     alfas=z_directions=np.arange(0               ,    2*np.pi     ,    np.pi/(cfg.RAY_CASTING_IMAGE_RESOLUTION/2)) # 256 values
     betas=y_directions=np.arange(-np.pi/2,    np.pi/2     ,    np.pi/cfg.RAY_CASTING_IMAGE_RESOLUTION) #  256 values
     unit_vector=[1,0,0]
     ray_directions={}
     for i in range(alfas.shape[0]):
         if i not in ray_directions:
             ray_directions[i]={}
         for j in range(betas.shape[0]):
                   alfa=z_directions[i]
                   beta=y_directions[j]
                   gamma=0
                   rotation_around_z= [  [ np.cos(alfa) , -np.sin(alfa), 0 ],  [ np.sin(alfa), -np.cos(alfa), 0 ],  [ 0 , 0, 1] ]
                   rotation_around_y= [  [ np.cos(beta) ,0, np.sin(beta)],  [0,1,0 ],  [-np.sin(beta) , 0, np.cos(beta)] ]
                   rotation_around_x= [  [1,0,0], [ 0,np.cos(gamma) , -np.sin(gamma) ],  [ 0, np.sin(gamma), np.cos(gamma) ] ]
                   rotation=np.matmul(rotation_around_z,np.matmul(rotation_around_y,rotation_around_x))
                   #yaw=rotation_around_z= [  [ cos_alfa , - sin_alfa, 0 ],  [ sin_alfa , - cos_alfa, 0 ],  [ 0 , 0, 1] ]
                   #pitch=rotation_around_y= [  [ cos_beta , 0,  sin_beta],  [ 0, 1, 0 ],  [ -sin_beta , 0, cos_beta] ].
                   #roll=rotation_around_x= [ [1,0,0 ], [0, cos_gamma,-sin_gamma], [0,sin_gamma,cos_gamma]]
                   #R=rotation_around_z * rotation_around_y * rotation_around_x
                   ray_direction=np.matmul(unit_vector,rotation)
                   ray_directions[i][j]=ray_direction

     return ray_directions


ray_directions=generate_ray_directions() 
i=0
for folder in mesh_folders:
 if folder.startswith(folderstart) :
    t1=time.time()
    mesh_path = folder +"/" + folder +".obj"
    RIR_folder  = path + "/" +folder +"/hybrid"
    i=i+1
    clear_scene()
    bmesh_object=bpy.ops.wm.obj_import(filepath= path + "/" +folder +"/" + folder +".obj")
    bpy.context.view_layer.update()
    if(os.path.exists(RIR_folder)):
        json_path = RIR_folder +"/sim_config.json"
        json_file = open(json_path)
        data = json.load(json_file)
        # receivers = len(data['receivers'])

        # if(receivers<(num_counter+temp_counter)):
        #     num_receivers =receivers #len(data['receivers'])
        #     temp_counter = temp_counter + (num_counter - receivers)
        # else:
        #     num_receivers = num_counter+temp_counter
        #     temp_counter = 0

        num_receivers = len(data['receivers'])
        num_sources = len(data['sources'])

        #print("num_receivers  ", num_receivers,"   num_sources  ", num_sources)
        for n in range(num_receivers):
            receiver = data['receivers'][n]['xyz']
            mesh_embedding_receiver_image = ray_cast(bmesh_object,receiver)
            mesh_embedding_receiver_image = mesh_embedding_receiver_image.detach().to(device='cuda:'+CUDANO)
            with torch.no_grad():
               mesh_embedding_receiver_image_latents=image_vae.encode(mesh_embedding_receiver_image).latent_dist.sample().reshape(int(4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))
            for s in range(num_sources):
                source = data['sources'][s]['xyz']
                RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
                RIR_path = folder +"/hybrid/" + RIR_name
                full_RIR_path = path+'/'+ RIR_path
                mesh_embedding_source_image=ray_cast(bmesh_object,source)
                mesh_embedding_source_image = mesh_embedding_source_image.detach().to(device='cuda:'+CUDANO)
                with torch.no_grad():
                   mesh_embedding_source_image_latents=image_vae.encode(mesh_embedding_source_image).latent_dist.sample().reshape(int(4*cfg.RAY_CASTING_IMAGE_RESOLUTION/8*cfg.RAY_CASTING_IMAGE_RESOLUTION/8))
                if(os.path.exists(full_RIR_path)):
                                  embed_data = {}
                                  embed_data["RIR"] =  get_RIR(os.path.join(path,RIR_path))
                                  embed_data["source_and_receiver"] =  np.concatenate((np.array(source).astype('float32'),np.array(receiver).astype('float32')))
                                  mesh_embed=torch.concatenate((mesh_embedding_source_image_latents,mesh_embedding_receiver_image_latents),axis=0).detach().cpu()
                                  embed_data["mesh_embedding"] = mesh_embed
                                  if folder.startswith('f') or folder.startswith('e') or folder.startswith('d') : # each having 300 , 900/5000 makes 18 percent of data will be validation data.
                                     validation_embedding_list.append(embed_data)
                                  else:
                                     training_embedding_list.append(embed_data)

    t2=time.time()
    print(f"{folder} {i}/{len(mesh_folders)} time:{t2-t1}",flush=True)




print("validation_embedding_list", len(validation_embedding_list))
print("training_embedding_list", len(training_embedding_list))

filler = 128  - (len(validation_embedding_list) % 128)
len_embed_list = len(validation_embedding_list) -1
if(filler < 128):
    for i in range(filler):
        validation_embedding_list.append(validation_embedding_list[len_embed_list-filler+i])

filler = 128  - (len(training_embedding_list) % 128)
len_embed_list = len(training_embedding_list) -1
if(filler < 128):
    for i in range(filler):
        training_embedding_list.append(training_embedding_list[len_embed_list-filler+i])

# embed_count = 128*2
# embedding_list = embedding_list[0:embed_count]
# print("embdiing_list12345", len(embedding_list))

training_embeddings_pickle =path+"/training.embeddings."+folderstart+".pickle"
with open(training_embeddings_pickle, 'wb') as f:
    pickle.dump(training_embedding_list, f, protocol=2)

validation_embeddings_pickle =path+"/validation.embeddings."+folderstart+".pickle"
with open(validation_embeddings_pickle, 'wb') as f:
    pickle.dump(validation_embedding_list, f, protocol=2)





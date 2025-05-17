import time
import torch.utils.data as data
import os
import numpy as np
import torch
import librosa
import sys
import bpy
import bmesh
from miscc.config import cfg
from PIL import Image
from contextlib import redirect_stdout
import io
import pickle

class RIRDataset(data.Dataset):
    def __init__(self,data_dir,embeddings,split='train',rirsize=4096): 
        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.embeddings = embeddings
        self.ray_directions=self.generate_ray_directions() 
        self.bpy_context = bpy.context
        self.bpy_scene = self.bpy_context.scene
        self.bpy_depsgraph=self.bpy_context.evaluated_depsgraph_get()
        bpy.context.scene.cycles.device = 'GPU'
    def get_RIR(self, full_RIR_path):

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

        resample_length = int(self.rirsize)
        
        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        with open(picklePath, 'wb') as f:
            pickle.dump(RIR, f, protocol=2)

        return RIR

    def __getitem__(self, index):

        graph_path,RIR_path,source_location,receiver_location= self.embeddings[index]
        data = {}
        data["RIR"] =  self.get_RIR(os.path.join(self.data_dir,RIR_path))
        data["source_and_receiver"] =  np.concatenate((np.array(source_location).astype('float32'),np.array(receiver_location).astype('float32')))
        data["mesh_embeddings_source_image"],data["mesh_embeddings_receiver_image"] = self.mesh_embeddings(os.path.join(self.data_dir,graph_path),source_location,receiver_location)
        return data
        
    def __len__(self):
        return len(self.embeddings)

    def mesh_embeddings(self,full_graph_path,source,receiver):
        self.clear_scene()
        bmesh_object=bpy.ops.wm.obj_import(filepath=full_graph_path)
        ray_cast_image_source=self.ray_cast(bmesh_object,source)
        ray_cast_image_receiver=self.ray_cast(bmesh_object,receiver)
        return ray_cast_image_source,ray_cast_image_receiver

    def clear_scene(self):
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

    def ray_cast(self,bmesh_object,origin):
       origin=list(np.array(origin).astype(np.float32))
       ray_casting_image=np.zeros((len(self.ray_directions.keys()),len(self.ray_directions[0].keys())))
       for alfa in self.ray_directions:
        for beta in self.ray_directions[alfa]:
          direction=self.ray_directions[alfa][beta]
          hit, loc, normal, idx, obj, mw = self.bpy_scene.ray_cast(self.bpy_depsgraph,origin, direction)
          if hit:
              distance=np.linalg.norm(np.array(origin)-np.array(loc))
              gray_scale_color=min(int(63+192*distance/cfg.MAX_RAY_CASTING_DISTANCE),255)
              ray_casting_image[alfa][beta]=gray_scale_color
#### NOT : MP : CALISMAZSA BIR DE NORMALI DE ISIN ICINE KATMAYI DENEYEBILIRIZ              
              #print(f"HIT: location={np.array(loc).shape} normal_vector_of_hit_point={np.array(norm).shape} idx={np.array(idx).shape} obj={np.array(obj).shape} mw={np.array(mw).shape}")
          else:
              ray_casting_image[alfa][beta]=0
       ray_casting_image=ray_casting_image.reshape(cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION,1).repeat(3,axis=2)
       ray_casting_image=Image.fromarray(np.uint8(ray_casting_image), mode="RGB")
       #ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32).reshape(1,3,cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION)
       ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32)

       return ray_casting_image


    def generate_ray_directions(self):
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



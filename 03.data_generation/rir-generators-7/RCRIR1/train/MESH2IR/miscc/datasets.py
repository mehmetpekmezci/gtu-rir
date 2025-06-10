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
import trimesh
import pygame


class RIRDataset(data.Dataset):
    def __init__(self,data_dir,embeddings,split='train',rirsize=4096): 
        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.embeddings = embeddings

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
        data["mesh_embeddings_source_image"],data["mesh_embeddings_receiver_image"],data["room_depth"],data["room_width"],data["room_height"] = self.mesh_embeddings(os.path.join(self.data_dir,graph_path),source_location,receiver_location)
        room_dims=[data["room_depth"],data["room_width"],data["room_height"]]
        data["source_and_receiver"] =  np.concatenate((np.array(source_location).astype('float32'),np.array(receiver_location).astype('float32'),np.array(room_dims).astype('float32')))
        return data
        
    def __len__(self):
        return len(self.embeddings)

    def loadMeshAndProject(self,full_graph_path):
        mesh = trimesh.load_mesh(full_graph_path)
        path3d= mesh.section(plane_origin=[0,0,0], plane_normal=[0, 1, 0]) 
        path2d,matrix_to_3D = path3d.to_planar()
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
        return path2d,DEPTH,WIDTH,HEIGHT,max_dim
    
    def generate_ray_cast_image(self,full_graph_path,path2d,DEPTH,WIDTH,origin,MESH_EXPAND_RATIO):
            #https://github.com/RaubCamaioni/Raycast-Shadows-
            SAVE_IMAGE=False
            
            #width, height = WIDTH*1.2,DEPTH*1.2
            width, height = WIDTH,DEPTH
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Visual Demo")
            segments=segments_from_path2d(path2d,WIDTH,DEPTH,MESH_EXPAND_RATIO)
            segments = np.array(segments)
            points = unique_points_from_segments(segments)
            screen.fill((255,255,255))
            rays = generate_rays_from_points(origin, points)
            intersections = raycast_rays_segments(rays, segments)
            closest = closest_intersection_from_raycast_rays_segments(intersections)
#            for origin2 in closest:  
#                    rays2=generate_rays_from_points(origin2, points)
#                    intersections2 = raycast_rays_segments(rays2, segments)
#                    closest2=closest_intersection_from_raycast_rays_segments(intersections2)
#                    pygame.draw.polygon(screen, (0,0,255), closest2)
#                    for intersect2 in closest2:
#                          pygame.draw.aaline(screen, (0, 255, 255), origin2, (intersect2[0], intersect2[1]))
            pygame.draw.polygon(screen, (100,0,0), closest)
            for intersect in closest:
                  pygame.draw.aaline(screen, (255, 0, 0), origin, (intersect[0], intersect[1]))
            draw_segments(screen, segments)
            pygame.display.flip()
            
            if SAVE_IMAGE:
                 filename=full_graph_path+'.'+str(int(origin[0]))+'.'+str(int(origin[1]))+'.ray_cast_image.png'
                 #print(filename)
                 pygame.image.save( screen, filename )
     
            img1=np.array(pygame.surfarray.array3d(screen)).copy()
            image = np.zeros((cfg.RAY_CASTING_IMAGE_RESOLUTION, cfg.RAY_CASTING_IMAGE_RESOLUTION, 3))
            image[:img1.shape[0],:img1.shape[1],:]=img1[:,:,:]
            return image.astype(np.float32).transpose(2, 0, 1)
            
    def mesh_embeddings(self,full_graph_path,source,receiver):
        full_graph_path=full_graph_path.replace(".pickle",".obj")
        path2d,ROOM_DEPTH,ROOM_WIDTH,ROOM_HEIGHT,max_dim=self.loadMeshAndProject(full_graph_path)
        #mesh = trimesh.load_mesh(full_graph_path, process=False)
        #Y=np.max(mesh.vertices[:,2])/2
        #X=np.max(mesh.vertices[:,0])/2
        MESH_EXPAND_RATIO=cfg.RAY_CASTING_IMAGE_RESOLUTION/max_dim
        DEPTH=ROOM_DEPTH*MESH_EXPAND_RATIO
        WIDTH=ROOM_WIDTH*MESH_EXPAND_RATIO
        source=np.array(source).astype(np.float32)*MESH_EXPAND_RATIO 
        receiver=np.array(receiver).astype(np.float32)*MESH_EXPAND_RATIO 

        source[0]+=DEPTH/2 
        source[2]=-source[2]
        source[2]+=WIDTH/2 

        receiver[0]+=DEPTH/2 
        receiver[2]=-receiver[2] 
        receiver[2]+=WIDTH/2 
#
#        s12=[] ## sadece bu olabilir, hepsinde calisan durum.
#        s12.append(s[0])
#        s12.append(-s[2])
#        s12.append(s[1])
#            

        ray_cast_image_source=self.generate_ray_cast_image(full_graph_path,path2d,DEPTH,WIDTH,(source[2],source[0]),MESH_EXPAND_RATIO) # -z ==y
        ray_cast_image_receiver=self.generate_ray_cast_image(full_graph_path,path2d,DEPTH,WIDTH,(receiver[2],receiver[0]),MESH_EXPAND_RATIO) # -z ==y
        #print(f"ray_cast_image_source.shape={ray_cast_image_source.shape}")
        #print(f"ray_cast_image_source.shape={ray_cast_image_source.shape}")
        #print(f"ray_cast_image_receiver.shape={ray_cast_image_receiver.shape}")

        return ray_cast_image_source,ray_cast_image_receiver,ROOM_DEPTH,ROOM_WIDTH,ROOM_HEIGHT

 
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


def segments_from_path2d(path2d,WIDTH,DEPTH,MESH_EXPAND_RATIO):
        segments=[]
       
        #print(vertices)
        vertices=np.array(path2d.vertices)*MESH_EXPAND_RATIO
        vertices[:,1]=vertices[:,1]*-1
        vertices[:,0]+=(WIDTH/2)
        vertices[:,1]+=(DEPTH/2)

        for line in path2d.entities:
                line_segments=vertices[line.points]
                #print(f"line_segments={line_segments}")
                if line_segments.shape[0]>2:
                     for i in range(1,line_segments.shape[0]-1):
                              segments.append(line_segments[i-1:i+1])
                else:
                     segments.append(line_segments)
        return segments

def draw_segments(screen, segments):
    for p1, p2 in segments:
        pygame.draw.line(screen, (0,0,0), p1, p2, 1)



import importlib
import math
import glob   
import sys 
import os        
import argparse  
import numpy as np        
import librosa   
import wave     
import scipy       
import scipy.io.wavfile
import scipy.fft
from scipy import signal
from scipy import stats
from scipy.spatial import distance
import pickle  
import matplotlib 
import matplotlib.pyplot as plt
import librosa
import librosa.display
import traceback
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchaudio
import torch
import torch.nn.functional as TF
import csv
import configparser
import trimesh

np.set_printoptions(threshold=sys.maxsize)

   # micNo
   #
   #    5          1
   #    |          |
   # 4-------||-------0
   #    |    ||    |
   #    6    ||    2                    


   # physicalSpeakerNo
   #              
   # 3---2---||
   #         || \    
   #         ||   1                      
   #         ||     \                      
   #         ||      0                    


def load_data(record_dir):

    data = {}
    data['mesh_filename']='mesh.obj'
    data['mesh']=trimesh.load_mesh(record_dir+'/'+data['mesh_filename'])

    config = configparser.ConfigParser()
    config.read(record_dir+'/'+'properties.ini')

    CENT=100

    data['roomX']=float(config.get('all','room_depth'))/CENT
    data['roomY']=float(config.get('all','room_width'))/CENT
    sourceX=(float(config.get('all','speakerStandInitialCoordinateX'))+float(config.get('all','speakerRelativeCoordinateX')))/CENT
    sourceY=(float(config.get('all','speakerStandInitialCoordinateY'))+float(config.get('all','speakerRelativeCoordinateY')))/CENT
    sourceZ=(float(config.get('all','speakerRelativeCoordinateZ')))/CENT
    data['source']=[sourceX,sourceY,sourceZ]
    receiverX=(float(config.get('all','microphoneStandInitialCoordinateX'))+float(config.get('all','mic_RelativeCoordinateX')))/CENT
    receiverY=(float(config.get('all','microphoneStandInitialCoordinateY'))+float(config.get('all','mic_RelativeCoordinateY')))/CENT
    receiverZ=(float(config.get('all','mic_RelativeCoordinateZ')))/CENT
    data['receiver']=[receiverX,receiverY,receiverZ]

    #microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ=permuteGANInput(microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ,roomDepth,roomWidth,roomHeight)

    return data



def origin_to_center_of_room(source,receiver,roomX,roomY):
    source=[source[0]-roomX/2,source[1]-roomY/2,source[2]]
    receiver=[receiver[0]-roomX/2,receiver[1]-roomY/2,receiver[2]]
    return source,receiver

def mesh_origin_to_center_of_room(roomX,roomY,mesh):
    mesh.vertices[:][0]=mesh.vertices[:][0]-roomX/2
    mesh.vertices[:][1]=mesh.vertices[:][1]-roomY/2
    return trimesh.Trimesh(mesh.vertices,mesh.faces)



def permutations(x,y,z,roomX,roomY,permutationNo,negationNo):

    finalResult=[x,y,z]

    match negationNo:
        case 0 : x=-x
        case 1 : y=-y
        case 2 : x=-x;y=-y
        case 3 : x=roomX-x
        case 4 : y=roomY-y
        case 5 : x=roomX-x;y=roomY-y

    match permutationNo:
        case 0 : finalResult = [x,y,z]
        case 1 : finalResult = [x,z,y]
        case 2 : finalResult = [y,x,z]
        case 3 : finalResult = [y,z,x]
        case 4 : finalResult = [z,x,y]
        case 5 : finalResult = [z,y,x]

    return finalResult

def save_record(record_dir,names,sources,receivers,mesh_filenames):
    with open(record_dir+'/newdata.csv', 'w', newline='') as csvfile:
       csvwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
       for i in range(len(names)):
          csvwriter.writerow([names[i], sources[i][0],sources[i][1],sources[i][2],receivers[i][0],receivers[i][1],receivers[i][2], mesh_filenames[i]])

def main(record_dir):
  data=load_data(record_dir)

  mesh=data['mesh']
  mesh_filename=data['mesh_filename']
  source=data['source']
  receiver=data['receiver']
  roomX=data['roomX']
  roomY=data['roomY']
  newmesh= mesh_origin_to_center_of_room(roomX,roomY,mesh)
  newmesh_filename='newmesh.obj'
  newmesh.export(record_dir+'/'+newmesh_filename)
  names=[]
  sources=[]
  receivers=[]
  meshes=[]
  for i in range(6):
    for j in range(6):
       names.append(f"{i}-{j}.0")
       newsource=permutations(source[0],source[1],source[2],roomX,roomY,i,j)
       sources.append(newsource)
       newreceiver=permutations(receiver[0],receiver[1],receiver[2],roomX,roomY,i,j)
       receivers.append(newreceiver)
       meshes.append(mesh_filename)

       names.append(f"{i}-{j}.1")

       newsource,newreceiver=origin_to_center_of_room(newsource,newreceiver,roomX,roomY)
       sources.append(newsource)
       receivers.append(newreceiver)
       meshes.append(newmesh_filename)
       
  save_record(record_dir,names,sources,receivers,meshes)


if __name__ == '__main__':
 main(str(sys.argv[1]).strip())


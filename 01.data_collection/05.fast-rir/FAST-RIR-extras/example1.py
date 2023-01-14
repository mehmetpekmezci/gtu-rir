import os
import numpy as np
import random
import argparse
import pickle

normalize_geometry_embeddings_list =[]

'''
Listener Position = LP
Source Position = SP
Room Dimension = RD
Reverberation Time = T60
Correction = CRR

CRR = 0.1 if 0.5<T60<0.6
CRR = 0.2 if T60>0.6
CRR = 0 otherwise

Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_Y,SP_Z,RD_X,RD_Y,RD_Z,(T60+CRR)] /5) - 1


for n in range(960):

	lx = (10/960)*n + 0.5
	geometry_embeddings= [lx,10.5,2.5,8.8,3.5,1.5,9,7,3,0.75]
	max_dimension = 5
	print(geometry_embeddings)
	normalize_geometry_embeddings =np.divide(geometry_embeddings,max_dimension)-1
	print(normalize_geometry_embeddings)
	normalize_geometry_embeddings_list.append(normalize_geometry_embeddings)

for LX in range(3):
 lx = (10/3)*LX + 0.5
 for LY in range(3):
  ly = (10/3)*LY + 0.5
  for LZ in range(3):
   lz = (3/3)*LZ + 0.2
   for MX in range(2):
    mx = (lx/2)*MX + 0.1
    for MY in range(2):
     my = (ly/2)*MY + 0.1
     for SX in range(2):
      sx = (lx/2)*SX + 0.1
      for SY in range(2):
       sy = (ly/2)*SY + 0.1
       for RT60 in range(2):
        rt60 = (1/2)*RT60 + 0.1
        geometry_embeddings= [lx,ly,lz,mx,my,2,sx,sy,2,rt60]
        max_dimension = 5
        print(geometry_embeddings)
        normalize_geometry_embeddings =np.divide(geometry_embeddings,max_dimension)-1
        print(normalize_geometry_embeddings)
        normalize_geometry_embeddings_list.append(normalize_geometry_embeddings)



#for LX in [6,8,12]:
for LX in [8,9,10,11]:
 lx = LX 
# for LY in [6,8,12]:
 for LY in [6,7,8]:
  ly = LY 
#  for LZ in [2.8,3.5]:
  for LZ in [2.5,3.0,3.5]:
   lz = LZ 
#   for MX in [6,8,10]:
   for MX in [2,6,10]:
    mx = MX 
#    for MY in [2,5,8]:
    for MY in [2,6,10]:
     my = MY + 0.1
     for SX in range(2,10):
      #sx = (MX+1.5+SX)%LX
#     for SX in [8,9,10]:
        #sx = (MX+SX)%LX
        sx=SX
#        for SY in [6,7,9]:
        for SY in range(2,8):
          sy = SY+0.1
#          for RT60 in [0.25,0.5,0.75]:
          for RT60 in [0.3,0.4,0.5,0.6,0.7]:



'''



 
for ROOM_DEPTH in [6,9,11]:
 room_depth = ROOM_DEPTH 
 for ROOM_WIDTH in [8,10,11]:
  room_width = ROOM_WIDTH
  for LZ in [2.5,3,3.5]:
   room_height = ROOM_HEIGHT
   for MX in [7,9,10]:
    mx = MX 
    for MY in [3,4,5]:
     my = MY 
     for SX in  [8,9,10]:
        sx = SX
        for SY in  [6,7,8]:
          sy = SY
          for RT60 in [0.5,0.75,0.9]:
            rt60 = RT60
	    #geometry_embeddings= [lx,10.5,2.5,8.8,3.5,1.5,9,7,3,0.75]
            #Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_Y,SP_Z,RD_X,RD_Y,RD_Z,(T60+CRR)] /5) - 1
            geometry_embeddings= [mx,my,lz,sx,sy,2,room_depth,room_width,room_height,rt60]
            max_dimension = 5
            print(geometry_embeddings)
            normalize_geometry_embeddings =np.divide(geometry_embeddings,max_dimension)-1
            print(normalize_geometry_embeddings)
            normalize_geometry_embeddings_list.append(normalize_geometry_embeddings)





embeddings_pickle ="example1.pickle"
with open(embeddings_pickle, 'wb') as f:
    pickle.dump(normalize_geometry_embeddings_list, f, protocol=2)

print(len(normalize_geometry_embeddings_list))

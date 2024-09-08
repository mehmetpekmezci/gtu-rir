#!/usr/bin/env python

import numpy as np
import trimesh
import sys
from furniture import prepare_furnitures 
from floor_plan import generate_floor_plans
from mesh_floor import generate_mesh,save_mesh
 
## DEFAULT METRIC is METRES


# python3 main.py 10 15 
FLAT_WIDTH=int(sys.argv[1]) # 10
FLAT_DEPTH=int(sys.argv[2]) # 15
FLAT_HEIGHT=float(sys.argv[3]) # 15
NUMBER_OF_SAMPLES=sys.argv[4]
DATA_DIR=sys.argv[5]


FLAT_FLOOR_HEIGHT=0.2
FURNITURES=prepare_furnitures()
## First arrange rooms in the flat by taking the corridor as referrence.
## Then arrannge the furnitures in the rooms.
FLOOR_PLANS=generate_floor_plans(FLAT_WIDTH,FLAT_DEPTH,FLAT_HEIGHT,FURNITURES,NUMBER_OF_SAMPLES,DATA_DIR)

for floor_plan in FLOOR_PLANS:
    generated_mesh=generate_mesh(floor_plan)
    save_mesh(DATA_DIR,floor_plan['id'],generated_mesh)

        
'''
statistics={}
statistics["BATH_ROOM"]={}
statistics["LIVING_ROOM"]={}
statistics["BED_ROOM"]={}
statistics["KITCHEN"]={}
statistics["HALL"]={}

for floor_plan in FLOOR_PLANS:
    for i in range(len(floor_plan['rooms'])):
      for j in range(len(floor_plan['rooms'][i])):
        room=floor_plan["rooms"][i][j]
        statistics[room["type"}]["area"]=""

'''

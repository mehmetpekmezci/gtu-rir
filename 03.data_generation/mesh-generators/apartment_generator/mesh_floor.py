#!/usr/bin/env python

import pymeshlab as ml
import numpy as np
import trimesh
import sys
import random
from furniture import prepare_furnitures , get_furniture_mesh
from random import randrange

## DEFAULT METRIC is METRES

def rotateToMesh2IRNormal(obj):
    angle=np.pi/2
    direction=[-1,0,0]
    center=[0,0,0]
    rotation = trimesh.transformations.rotation_matrix(angle,direction,center)
    obj.apply_transform(rotation)
    
    
def generate_mesh(floor_plan):

    FLAT_FLOOR_THICKNESS=0.2
    
    scene = trimesh.Scene()
    
    # floor plane
    plane = trimesh.creation.box(extents=[floor_plan['dimensions']['width'],floor_plan['dimensions']['depth'], FLAT_FLOOR_THICKNESS])
    scene.add_geometry(plane)

    apartment_box = trimesh.creation.box(extents=[floor_plan['dimensions']['width'],floor_plan['dimensions']['depth'],floor_plan['dimensions']['height']+0.01])
    apartment_box.apply_translation([0, 0, floor_plan['dimensions']['height']/2+0.01+ FLAT_FLOOR_THICKNESS] )
    scene.add_geometry(apartment_box)
    #print(f"\n### APARTMENT NO: {floor_plan['id']} #####################################\n")
    for i in range(len(floor_plan['rooms'])):
      for j in range(len(floor_plan['rooms'][i])):
        room=floor_plan["rooms"][i][j]
        if room["type"] != "HALL":
           #print(f"room:{room['name']}")
           room_width=room["coordinates"]["x"]["end"]["value"]-room["coordinates"]["x"]["start"]["value"]
           room_depth=room["coordinates"]["y"]["end"]["value"]-room["coordinates"]["y"]["start"]["value"]
        
           room_box=trimesh.creation.box(extents=[room_width,room_depth,floor_plan['dimensions']['height']])
           scene.add_geometry(room_box)
           roomX=(-floor_plan['dimensions']['width']+room_width)/2+room["coordinates"]["x"]["start"]["value"]
           roomY=(-floor_plan['dimensions']['depth']+room_depth)/2+room["coordinates"]["y"]["start"]["value"]
           roomZ=(floor_plan['dimensions']['height']+FLAT_FLOOR_THICKNESS)/2
           
           room_box.apply_translation([roomX, roomY, roomZ ] )

           
           for furniture in room["furnitures"]:
               
               dimX=furniture["dimensions"][0]
               dimY=furniture["dimensions"][1]
               dimZ=furniture["dimensions"][2]
               i_start=furniture["start_region"]["i"]/10
               i_end=furniture["end_region"]["i"]/10
               j_start=furniture["start_region"]["j"]/10
               j_end=furniture["end_region"]["j"]/10
               
               if abs((i_end-i_start) - dimX) > 0.1:
                   #print(f'Turning {furniture["name"]} : dimX={dimX} dimY={dimY} dimZ={dimZ} i_start={i_start}  i_end={i_end}  j_start={j_start}  j_end={j_end}  ')
                   temp=furniture["dimensions"][0]
                   furniture["dimensions"][0]=furniture["dimensions"][1]
                   furniture["dimensions"][1]=temp

               dimX=furniture["dimensions"][0]
               dimY=furniture["dimensions"][1]
               dimZ=furniture["dimensions"][2]
               
               furnitureMesh=get_furniture_mesh(furniture,FLAT_FLOOR_THICKNESS)
               
               scene.add_geometry(furnitureMesh)
               #print(f'{furniture["name"]} : dimX={dimX} dimY={dimY} dimZ={dimZ} i_start={i_start}  i_end={i_end}  j_start={j_start}  j_end={j_end}  ')
               furnitureMesh.apply_translation([roomX-room_width/2+dimX/2+i_start,roomY-room_depth/2+dimY/2+j_start, 0 ] )

               





    mesh = trimesh.util.concatenate(scene.dump())
    #mesh.visual = trimesh.visual.ColorVisuals()
    rotateToMesh2IRNormal(mesh)
    return mesh       

'''                
total_=0
for room_type in FURNITURES:
    for name in FURNITURES[room_type]:
        print(f"{room_type}/{name}")
        mesh=FURNITURES[room_type][name]["mesh"]
        scene.add_geometry(FURNITURES[room_type][name]["mesh"])
        translation = [ 0,total_, 0] 
        mesh.apply_translation(translation)
        total_+=FURNITURES[room_type][name]["dimensions"][1]+0.5
        total_+=FURNITURES[room_type][name]["dimensions"][0]+0.5
        
        
# object-1 (box)
ROOM_DIM=[4,4,2.5]
box = trimesh.creation.box(extents=[ROOM_DIM[0],ROOM_DIM[1], ROOM_DIM[2]])
box.visual.face_colors = [0, 1., 0, 0.5]
translation =   # box offset + plane offset
box.apply_translation(translation)


    scene.add_geometry(box)



'''


    
    
def save_mesh(DATA_DIR,floor_plan_id,mesh):
   
    file_path=DATA_DIR+f'/floor_plan-{floor_plan_id}.mesh.obj'

    MESH2IR_VGAE_MESH_INPUT_FACE_SIZE=64*(28+randrange(16))
    #MESH2IR_VGAE_MESH_INPUT_FACE_SIZE=2000
    #MESH2IR_VGAE_MESH_INPUT_FACE_SIZE=1000
    #MESH2IR_VGAE_MESH_INPUT_FACE_SIZE=700

    face_index=list(range(len(mesh.faces)))
    print(f"len(face_index)={len(face_index)}")
    random.shuffle(face_index)
    face_index=face_index[:int(len(face_index)/2)]      
    print(f"len(face_index)={len(face_index)}")

    while mesh.faces.shape[0] < MESH2IR_VGAE_MESH_INPUT_FACE_SIZE :
        print(mesh.faces.shape[0])
        mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces,face_index=face_index)

    print('--output mesh has ', mesh.vertices.shape[0], ' vertex and ', mesh.faces.shape[0], ' faces')
    mesh.export(file_path,include_color=True)


    ms = ml.MeshSet()
    ms.load_new_mesh(file_path)
    m = ms.current_mesh()
    #ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=MESH2IR_VGAE_MESH_INPUT_FACE_SIZE, preservenormal=True)
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=MESH2IR_VGAE_MESH_INPUT_FACE_SIZE, preservenormal=True)
    print("Decimated to ", MESH2IR_VGAE_MESH_INPUT_FACE_SIZE, " faces mesh has ", ms.current_mesh().vertex_number(), " vertex")

    m = ms.current_mesh()
    print('output mesh has ', m.vertex_number(), ' vertex and ', m.face_number(), ' faces')
    ms.save_current_mesh(file_path)




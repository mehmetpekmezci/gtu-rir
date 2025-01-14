#!/usr/bin/env python

import numpy as np
import sys
import pymeshlab as ml
import trimesh
import random


from data import get_gtu_room_data




# DEPTH X
# WIDTH Y
# HEIGHT Z

 
## DEFAULT METRIC is METRES

DATA_DIR=sys.argv[1]

FURNITURE_TYPE,GTU_ROOM=get_gtu_room_data()

FLAT_FLOOR_THICKNESS=0.02

## DEFAULT METRIC is METRES


def rotateObject(obj,direction,angle):
    
    center=[0,0,0]
    rotation = trimesh.transformations.rotation_matrix(angle,direction,center)
    obj.apply_transform(rotation)

    
def rotateToMesh2IRNormal(obj):
    direction=[-1,0,0]
    rotateObject(obj,direction,np.pi/2)



        
def get_box_mesh(box_shape,name):
     mesh=trimesh.creation.box(extents=[box_shape["DEPTH"],box_shape["WIDTH"],box_shape["HEIGHT"]], metadata={"name":name})
     
     TRANSLATION_X=0
     TRANSLATION_Y=0
     TRANSLATION_Z=0
     
     if "DEPTH_FROM_BACK" in box_shape:
         TRANSLATION_X= box_shape["DEPTH_FROM_BACK"]
     if "WIDTH_FROM_BACK" in box_shape:
         TRANSLATION_Y= box_shape["WIDTH_FROM_BACK"]
     if "HEIGHT_FROM_GROUND" in box_shape:
         TRANSLATION_Z= box_shape["HEIGHT_FROM_GROUND"]
         
     translation = [ TRANSLATION_X, TRANSLATION_Y, TRANSLATION_Z]    
     mesh.apply_translation(translation)
     return mesh

def get_cylinder_mesh(radius,height,X,Y,name):
     mesh=trimesh.creation.cylinder(radius=radius, height=height, metadata={"name": name})
     translation = [X,Y,0]    
     mesh.apply_translation(translation)
     return mesh

def get_chair_mesh(chair,roomX,roomY,rotationAngle,roomDepth,roomWidth):
     mesh_array=[]
     
     #chair["WIDTH"]=53
     #chair["DEPTH"]=9+59
     #chair["HEIGHT"]=77
     mesh_array.append(get_box_mesh(chair["OTURAK"],"CHAIR_SEAT"))
     mesh_array.append(get_box_mesh(chair["ARKA"],"CHAIR_BACK"))
     mesh_array.append(get_box_mesh(chair["TABLA"],"CHAIR_TABLE"))
     mesh_array.append(get_cylinder_mesh(0.02,chair["OTURAK"]["HEIGHT_FROM_GROUND"],0,0,"CHAIR_LEG_1"))
     mesh_array.append(get_cylinder_mesh(0.02,chair["OTURAK"]["HEIGHT_FROM_GROUND"],chair["OTURAK"]["DEPTH"],0,"CHAIR_LEG_2"))
     mesh_array.append(get_cylinder_mesh(0.02,chair["OTURAK"]["HEIGHT_FROM_GROUND"],0,chair["OTURAK"]["WIDTH"],"CHAIR_LEG_3"))
     mesh_array.append(get_cylinder_mesh(0.02,chair["OTURAK"]["HEIGHT_FROM_GROUND"],chair["OTURAK"]["DEPTH"],chair["OTURAK"]["WIDTH"],"CHAIR_LEG_4"))

     
     for i in range(len(mesh_array)):
         mesh=mesh_array[i]
         direction=[0,0,1]
         rotateObject(mesh,direction,rotationAngle)
         mesh.apply_translation([roomX,roomY,0])
         if i==0: mesh.apply_translation([-roomDepth/2+chair["WIDTH"]/2,-roomWidth/2+chair["DEPTH"]/2,0] )
         elif i==1: mesh.apply_translation([-roomDepth/2+chair["WIDTH"]/2,-roomWidth/2+chair["OTURAK"]["DEPTH"]/4,chair["ARKA"]["HEIGHT"]/2] )
         elif i==2: mesh.apply_translation([-roomDepth/2+chair["WIDTH"],-roomWidth/2+chair["DEPTH"]/2,0])
         else: mesh.apply_translation([-roomDepth/2+chair["WIDTH"]*3/4,-roomWidth/2+chair["OTURAK"]["DEPTH"]/4,chair["OTURAK"]["HEIGHT_FROM_GROUND"]/2] )

     return mesh_array
     
     
def get_table_mesh(table,roomX,roomY,rotationAngle,roomDepth,roomWidth):
     mesh_array=[]
     mesh_array.append(get_box_mesh(table,"TABLE"))
     mesh_array.append(get_cylinder_mesh(0.02,table["HEIGHT_FROM_GROUND"],0,0,"TABLE_LEG_1"))
     mesh_array.append(get_cylinder_mesh(0.02,table["HEIGHT_FROM_GROUND"],table["DEPTH"],0,"TABLE_LEG_2"))
     mesh_array.append(get_cylinder_mesh(0.02,table["HEIGHT_FROM_GROUND"],0,table["WIDTH"],"TABLE_LEG_3"))
     mesh_array.append(get_cylinder_mesh(0.02,table["HEIGHT_FROM_GROUND"],table["DEPTH"],table["WIDTH"],"TABLE_LEG_4"))

     for mesh in mesh_array:
         direction=[0,0,1]
         rotateObject(mesh,direction,rotationAngle)
         mesh.apply_translation([roomX,roomY,0])
         if mesh!=mesh_array[0]: mesh.apply_translation([-roomDepth/2+table["WIDTH"]/2,-roomWidth/2+table["DEPTH"]/2,table["HEIGHT_FROM_GROUND"]/2] )
         else: mesh.apply_translation([-roomDepth/2+table["WIDTH"],-roomWidth/2,0] )
     return mesh_array
     
     
def get_konsol_mesh(konsol,roomX,roomY,rotationAngle,roomDepth,roomWidth):
     mesh_array=[]
     mesh_array.append(get_box_mesh(konsol,"KONSOL"))
     for mesh in mesh_array:
         direction=[-1,0,0]
         rotateObject(mesh,direction,rotationAngle)
         mesh.apply_translation([roomX,roomY,0])
         mesh.apply_translation([-roomDepth/2+konsol["DEPTH"]/2,-roomWidth/2+konsol["WIDTH"]/2,konsol["HEIGHT"]/2] )
     return mesh_array
     
     

     
    
def generate_mesh(room_id):

    room=GTU_ROOM[room_id]

  
    scene = trimesh.Scene()
    
    # floor plane
    #plane = trimesh.creation.box(extents=[GTU_ROOM[room_id]["DEPTH"],GTU_ROOM[room_id]["WIDTH"], FLAT_FLOOR_THICKNESS])
    #scene.add_geometry(plane)

    apartment_box = trimesh.creation.box(extents=[GTU_ROOM[room_id]["DEPTH"],GTU_ROOM[room_id]["WIDTH"],GTU_ROOM[room_id]["HEIGHT"]+0.01])
    apartment_box.apply_translation([GTU_ROOM[room_id]["DEPTH"]/2, -GTU_ROOM[room_id]["WIDTH"]/2, GTU_ROOM[room_id]["HEIGHT"]/2] )
    scene.add_geometry(apartment_box)

 
    for key in room["FURNITURE_ARRAY"]:
       furniture_set=room["FURNITURE_ARRAY"][key]
       
       for j in range( furniture_set["COUNT"] ) :
##### BURADA FARKLI ODA TURLERINDE FARKLI DAVRANISLAR GEREKEBILIR , BUNA BAKMAK LAZIM
            roomX=furniture_set["X"]+ j* FURNITURE_TYPE[furniture_set["TYPE"]]['WIDTH']
#####            
            roomY=furniture_set["Y"]
            rotationAngle=furniture_set["ORIENTATION"]
            mesh_array=None
            if furniture_set["TYPE"] == "CHAIR" :
               mesh_array=get_chair_mesh(FURNITURE_TYPE["CHAIR"],roomX,roomY,rotationAngle,GTU_ROOM[room_id]["DEPTH"],GTU_ROOM[room_id]["WIDTH"])
            if furniture_set["TYPE"] == "KONSOLE1" :
               mesh_array=get_konsol_mesh(FURNITURE_TYPE["KONSOLE1"],roomX,roomY,rotationAngle,GTU_ROOM[room_id]["DEPTH"],GTU_ROOM[room_id]["WIDTH"])
            if furniture_set["TYPE"] == "KONSOLE2" :
               mesh_array=get_konsol_mesh(FURNITURE_TYPE["KONSOLE2"],roomX,roomY,rotationAngle,GTU_ROOM[room_id]["DEPTH"],GTU_ROOM[room_id]["WIDTH"])
            if furniture_set["TYPE"] == "TABLE" :
               mesh_array=get_table_mesh(FURNITURE_TYPE["TABLE"],roomX,roomY,rotationAngle,GTU_ROOM[room_id]["DEPTH"],GTU_ROOM[room_id]["WIDTH"])
            
            for mesh in mesh_array:
                mesh.apply_translation([GTU_ROOM[room_id]["DEPTH"]/2, -GTU_ROOM[room_id]["WIDTH"]/2, 0] )
               #mesh.apply_translation([GTU_ROOM[room_id]["DEPTH"]/2, -GTU_ROOM[room_id]["WIDTH"]/2, GTU_ROOM[room_id]["HEIGHT"]/2] )
            scene.add_geometry(mesh_array)
               
    mesh = trimesh.util.concatenate(scene.dump())
    #mesh.visual = trimesh.visual.ColorVisuals()
    rotateToMesh2IRNormal(mesh)
    #mesh.apply_translation([GTU_ROOM[room_id]["DEPTH"]/2, -GTU_ROOM[room_id]["WIDTH"]/2, GTU_ROOM[room_id]["HEIGHT"]/2] )
    #mesh.apply_translation([GTU_ROOM[room_id]["DEPTH"]/2, GTU_ROOM[room_id]["WIDTH"]/2, 0] )
    return mesh       




    
    
def save_mesh(DATA_DIR,room_id,mesh):
   
    file_path=DATA_DIR+f'/gtu-cs-room-{room_id}.mesh.obj'
    MESH2IR_VGAE_MESH_INPUT_FACE_SIZE=4000

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



room_id="208"
generated_mesh=generate_mesh(room_id)
save_mesh(DATA_DIR,room_id,generated_mesh)


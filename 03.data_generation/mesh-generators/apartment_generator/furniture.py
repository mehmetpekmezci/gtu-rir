import numpy as np
import trimesh
import sys
import numpy as np
import random
import copy
import math



          ## OBJ FILE COORDS ##                     ##  COORDS ON IMAGE - PNG ##
    ###################################         ###################################                                           
    # 2.0          2.1            2.2 #         # 0.0          0.1            0.2 #         
    #                                 #         #                                 #             
    #                                 #         #                                 #
    # 1.0          1.1            1.2 #         # 1.0          1.1            1.2 # 
    #                                 #         #                                 #
    #                                 #         #                                 #
    # 0.0          0.1            0.2 #         # 2.0          2.1            2.2 #       
    ###################################         ###################################  
    #                y                                                                   
    #            z  /                               ____ x                                               
    #            | /                                |                                      
    #            |/_____ x                          |
    #                                               y  


    
    
    
def define_furniture(name,room_type,leg_height,dimensions,dictionary,color,shapetype="box"):

    furniture={}
    furniture["dimensions"]=dimensions
    furniture["leg_height"]=leg_height     
    furniture["color"]=color     
    furniture["name"]=name     
    furniture["room_type"]=room_type   
    furniture["shape_type"]=shapetype  
    
    if room_type not in dictionary:
       dictionary[room_type]={}
    dictionary[room_type][name]=furniture
    
    

def get_furniture_mesh(furniture,FLAT_FLOOR_THICKNESS):    
    
    mesh=trimesh.creation.box(extents=[furniture["dimensions"][0],furniture["dimensions"][1],furniture["dimensions"][2]], metadata={"name": furniture["name"]})
    
    if furniture["shape_type"]=="cylinder":
         mesh=trimesh.creation.cylinder(radius=furniture["dimensions"][0]/2, height=furniture["dimensions"][2], metadata={"name": furniture["name"]})
       
    if furniture["leg_height"] > 0 :
       translation = [ 0, 0, furniture["leg_height"]+FLAT_FLOOR_THICKNESS/2]      
    else:
       translation = [ 0, 0, (furniture["dimensions"][2]+FLAT_FLOOR_THICKNESS)/2]    
    
    
    mesh.apply_translation(translation)
    
    return mesh
       




def prepare_furnitures():
    FURNITURES={} ## yatak, koltuk, masa, dolap,

    ##  KEY POINT IS :  ORDER : BIG AREA COMES FIRST.
    
    define_furniture("WRD3",		"BED_ROOM",	0,	[0.6,2.1,2],	FURNITURES,"#888888")#WARDROBE WITH 3 DOORS
    define_furniture("BED2",		"BED_ROOM",	0,	[2,2,0.7],	FURNITURES,"#222222")#BED for 2 PERSONs
    define_furniture("BED1",		"BED_ROOM",	0,	[1,2,0.7],	FURNITURES,"#666666")#BED for 1 PERSON
    define_furniture("WRD2",		"BED_ROOM",	0,	[0.6,1.2,2],	FURNITURES,"#AAAAAA")#WARDROBE WITH 2 DOORS
    define_furniture("TBLS",		"BED_ROOM",	0.7,	[0.6,0.9,0.02],	FURNITURES,"#CCCCCC")#SMALL STUDY TABLE
    define_furniture("WRD1",		"BED_ROOM",	0,	[0.6,0.6,2],	FURNITURES,"#DEDEDE")#WARDROBE WITH 1 DOOR

    define_furniture("CHR3",	"LIVING_ROOM",	0,	[0.8,2.4,0.4],	FURNITURES,"#002222")#ARM CHAIR FOR 3 PERSONs
    define_furniture("TBL",			"LIVING_ROOM",	0.7,	[0.9,1.6,0.02],	FURNITURES,"#004444")#DINING TABLE
    define_furniture("CHR2",	"LIVING_ROOM",	0,	[0.8,1.6,0.4],	FURNITURES,"#006666")#ARM CHAIR FOR 2 PERSONs
    define_furniture("CONS",			"LIVING_ROOM",	0,	[0.5,1.5,0.7],	FURNITURES,"#008888")#CONSOLE
    define_furniture("CHR1",	"LIVING_ROOM",	0,	[0.8,0.8,0.4],	FURNITURES,"#00AAAA")#ARM CHAIR FOR 1 PERSON
    define_furniture("TBLS",		"LIVING_ROOM",	0.7,	[0.6,0.9,0.02],	FURNITURES,"#00CCCC")#SMALL STUDY TABLR
    
    define_furniture("REFR",		"KITCHEN",	0,	[0.6,0.6,2],	FURNITURES,"#444400")#REFRIGIRATOR
    define_furniture("DISW",		"KITCHEN",	0,	[0.6,0.5,0.6],	FURNITURES,"#666600")#DISH WASHER
    define_furniture("DRWR",			"KITCHEN",	0,	[0.6,0.5,0.7],	FURNITURES,"#888800")#DRAWER
    define_furniture("TBL",			"KITCHEN",	0.7,	[0.9,1.6,0.02],	FURNITURES,"#222200")#DINING TABLE
    define_furniture("CAB",			"KITCHEN",	1.4,	[0.5,0.5,0.5],	FURNITURES,"#AAAA00")#CABINET
    
    define_furniture("SHWC",		"BATH_ROOM",	0,	[0.9,0.9,1.8],	FURNITURES,"#440044")#SHOWER CABIN
    define_furniture("WTRC",		"BATH_ROOM",	0,	[0.6,0.6,0.4],	FURNITURES,"#AA00AA",shapetype="cylinder")#WATER CLOSET
    define_furniture("WSHB",		"BATH_ROOM",	0,	[0.6,0.6,1.0],	FURNITURES,"#660066",shapetype="cylinder")#WASH BASIN
    define_furniture("DRWR",			"BATH_ROOM",	0,	[1,2,0.7],	FURNITURES,"#220022")#DRAWER
    define_furniture("CAB",			"BATH_ROOM",	1.4,	[0.5,0.5,0.5],	FURNITURES,"#880088")#CABINET
    
    return FURNITURES

    
def add_new_furniture_to_room(new_furniture,room):
    #print(f"############## {new_furniture['name']} add to {room['name']} #################")
    ## ALL THE ROOM IS SPLIT INTO 0.1 metre square regions.
    I=int(abs(room["coordinates"]["x"]["end"]["value"]-room["coordinates"]["x"]["start"]["value"])*10)
    J=int(abs(room["coordinates"]["y"]["end"]["value"]-room["coordinates"]["y"]["start"]["value"])*10)
    new_furniture_width=math.ceil(new_furniture["dimensions"][0]*10)
    new_furniture_depth=math.ceil(new_furniture["dimensions"][1]*10)
    #print(f"new_furniture_width:{new_furniture_width} -- new_furniture_depth:{new_furniture_depth} -- I:{I} J:{J}")

    room_matrix=np.zeros((I,J))
    
    #print(f"room_matrix.shape={room_matrix.shape}")
    
    ROOM_MIDDLE_SPACE_I=int(I/3)
    ROOM_MIDDLE_SPACE_J=int(J/3)
       
    for i in range(ROOM_MIDDLE_SPACE_I):
      for j in range(ROOM_MIDDLE_SPACE_J):
       if int(I/2)+i-(ROOM_MIDDLE_SPACE_I/2) < I and int(I/2)+i-(ROOM_MIDDLE_SPACE_I/2)>= 0 and int(J/2)+j-(ROOM_MIDDLE_SPACE_J/2) < J and int(J/2)+j-(ROOM_MIDDLE_SPACE_J/2)>=0 : 
         room_matrix[int(I/2+i-ROOM_MIDDLE_SPACE_I/2) ][int(J/2+j-ROOM_MIDDLE_SPACE_J/2) ]=3
     
     
    door_i=int(abs(room["door"]["x"]-room["coordinates"]["x"]["start"]["value"])*10)
    door_j=int(abs(room["door"]["y"]-room["coordinates"]["y"]["start"]["value"])*10)
   
    THRESHOLD=5
    for i in range(room_matrix.shape[0]):
     for j in range(room_matrix.shape[1]):
       if (room["door"]["orientation"] == "y" and abs(i-door_i)<THRESHOLD) or  (room["door"]["orientation"] == "x" and abs(j-door_j)<THRESHOLD):
           if door_i <= I/2 and door_j<J/2 and i <= I/2 and j <= J/2  : room_matrix[i][j]=2
           if door_i > I/2 and door_j<J/2 and i >= I/2 and j <= J/2  : room_matrix[i][j]=2  
           if door_i > I/2 and door_j>=J/2 and i >= I/2 and j >= J/2  : room_matrix[i][j]=2  
           if door_i <= I/2 and door_j>=J/2 and i <= I/2 and j >= J/2  : room_matrix[i][j]=2  

 
    '''
    for i in range(room_matrix.shape[0]):
     for j in range(room_matrix.shape[1]):
       if door_i <= I/2 and door_j<J/2 and i <= I/2 and j <= J/2  : room_matrix[i][j]=2  
       if door_i > I/2 and door_j<J/2 and i >= I/2 and j <= J/2 : room_matrix[i][j]=2  
       if door_i > I/2 and door_j>=J/2 and i >= I/2 and j >= J/2  : room_matrix[i][j]=2  
       if door_i <= I/2 and door_j>=J/2 and i <= I/2 and j >= J/2 : room_matrix[i][j]=2  
    '''

    
    for existing_furniture in room["furnitures"]:
        start_region_i=existing_furniture["start_region"]["i"]
        start_region_j=existing_furniture["start_region"]["j"]    
        end_region_i=existing_furniture["end_region"]["i"]
        end_region_j=existing_furniture["end_region"]["j"]
        #print(f"start_region_i = {start_region_i} , end_region_i = {end_region_i} , start_region_j = {start_region_j} , end_region_j = {end_region_j}")
        for i in range(start_region_i-3,end_region_i+3):
           for j in range(start_region_j-3,end_region_j+3):
               if i < I and i >=0 and j<J and j>=0:
                room_matrix[i][j]=1

       
    #print(room_matrix)
    found=False


    start_i,end_i,step_i=room_matrix.shape[0]-1,-1,-1
    start_j,end_j,step_j=room_matrix.shape[1]-1,-1,-1
    new_furniture_depth,new_furniture_width=new_furniture_depth,new_furniture_width
    
        
    if door_i <= I/2 and door_j<J/2 :
       start_i,end_i,step_i=room_matrix.shape[0]-1,-1,-1
       start_j,end_j,step_j=room_matrix.shape[1]-1,-1,-1
    if door_i > I/2 and door_j<J/2 :
       start_i,end_i,step_i=0,room_matrix.shape[0],1
       start_j,end_j,step_j=room_matrix.shape[1]-1,-1,-1
    if door_i > I/2 and door_j>=J/2 :
       start_i,end_i,step_i=0,room_matrix.shape[0],1
       start_j,end_j,step_j=0,room_matrix.shape[1],1
    if door_i <= I/2 and door_j>=J/2 :
       start_i,end_i,step_i=room_matrix.shape[0]-1,-1,-1
       start_j,end_j,step_j=0,room_matrix.shape[1],1
       
#    rand=random.randint(0,1)
#    if rand==0:


    for furniture_orientation in ["horizontal","vertical"]:
      for i in range(start_i,end_i,step_i):
       for j in range(start_j,end_j,step_j):
        if not found:
          if furniture_orientation == "horizontal" :
           found=search_and_place_the_furniture(room_matrix,new_furniture,room,i,j,new_furniture_depth,new_furniture_width,I,J)
          if furniture_orientation == "vertical" :
           found=search_and_place_the_furniture(room_matrix,new_furniture,room,i,j,new_furniture_width,new_furniture_depth,I,J)

 
    room["matrix"]=room_matrix
    



def search_and_place_the_furniture(room_matrix,new_furniture,room,i,j,new_furniture_width,new_furniture_depth,I,J):
         found=False
         if room_matrix[i][j]==0 and (i==0 or i==I-1 or j==0 or j==J-1):
            available=True
            #print(f"room_matrix.shape[0]={room_matrix.shape[0]} i={i} j={j} ")
            for i1 in range(1,new_furniture_width+1):
             #print(f"AAAAA room_matrix.shape[0]={room_matrix.shape[0]} i={i} i1={i1} room_matrix.shape[1]={room_matrix.shape[1]} j={j} ")
             if i+i1-1 >= room_matrix.shape[0]: available=False
             if available: 
              for j1 in range(1,new_furniture_depth+1):
                  #print(f"room_matrix.shape[0]={room_matrix.shape[0]} i={i} i1={i1} room_matrix.shape[1]={room_matrix.shape[1]} j={j} j1={j1}")
                  if j+j1-1 >= room_matrix.shape[1]: available=False
                  elif room_matrix[i+i1-1][j+j1-1]>0: available=False
            if available:
               new_furniture["start_region"]={}
               new_furniture["start_region"]["i"]=i
               new_furniture["start_region"]["j"]=j
               new_furniture["end_region"]={}
               new_furniture["end_region"]["i"]=i+new_furniture_width
               new_furniture["end_region"]["j"]=j+new_furniture_depth
               #print(f"appending furniture with region : {new_furniture['start_region']} -- {new_furniture['end_region']}")
               room["furnitures"].append(new_furniture)
               found=True          
         return found

def decorate_rooms(floor_plan,available_furnitures):

    ## available furnitures are already ordered by their floor area. ( biggest comes first)
    rooms=floor_plan["rooms"]
    for i in range(len(rooms)) :
      for j in range(len(rooms[i])):
        room=rooms[i][j]
        if room["type"] == "HALL" :
           continue
        room["furnitures"]=[]
        for furniture_name in available_furnitures[room["type"]]:
          #number_of_this_furniture_in_the_selected_room=random.randint(0,2)
          #for i in range(number_of_this_furniture_in_the_selected_room):
              room_s_furniture=copy.deepcopy(available_furnitures[room["type"]][furniture_name])
              add_new_furniture_to_room(room_s_furniture,room)
                   


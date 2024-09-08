import numpy as np
import trimesh
import sys
import pickle
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import random
import copy
import numpy as np
import math
from furniture import add_new_furniture_to_room,decorate_rooms


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

     
    
def create_single_room_definition(flat_width,flat_depth,room_name,room_type,room_color,room_dimension_minimum_length,room_dimension_maximum_length,room_region_x,room_region_y,MAX_I,MAX_J):

    room={}  

    room["name"]=room_name
    room["type"]=room_type
    room["color"]=room_color
    room["region"]={}
    room["region"]["x"]=room_region_x
    room["region"]["y"]=room_region_y
    room["min_len"]=room_dimension_minimum_length ## all dimensions of the room are grater than this value
    room["max_len"]=room_dimension_maximum_length ## all dimensions of the room are smaller than this value
    room["coordinates"]={}
    room["coordinates"]["x"]={}
    room["coordinates"]["y"]={}
    room["coordinates"]["x"]["start"]={}
    room["coordinates"]["x"]["start"]["value"]=room_region_x*(flat_width/2-room["min_len"]/2) 
    room["coordinates"]["x"]["start"]["growable"]=(room_region_x!=0) 
    room["coordinates"]["x"]["end"]={}
    room["coordinates"]["x"]["end"]["value"]=room_region_x*(flat_width/2)+(2-room_region_x)*room["min_len"]/2 
    room["coordinates"]["x"]["end"]["growable"]=(room_region_x!=MAX_I) # 2 = max i
    room["coordinates"]["y"]["start"]={}
    room["coordinates"]["y"]["start"]["value"]=room_region_y*(flat_depth/2-room["min_len"]/2)
    room["coordinates"]["y"]["start"]["growable"]=(room_region_y!=0)
    room["coordinates"]["y"]["end"]={}
    room["coordinates"]["y"]["end"]["value"]=room_region_y*(flat_depth/2)+(2-room_region_y)*room["min_len"]/2 
    room["coordinates"]["y"]["end"]["growable"]=(room_region_y!=MAX_J)



    return room
 
    
def create_random_room_definition(flat_width,flat_depth):

    # 0 1 2
    # 3 4 5
    # 6 7 8
    
    # 0.0 0.1 0.2
    # 1.0 1.1 1.2
    # 2.0 2.1 2.2
    

    rooms=[]
    for i in range(3):
        horizontal=[]
        rooms.append(horizontal)
        for j in range(3):
          room={} ## insert empty definition
          horizontal.append(room)
    
    MAX_I=len(rooms)-1
    MAX_J=len(rooms[0])-1  
    
    positions=list(range(9))    
    random.shuffle(positions)
    
    ENTERANCE_positions=[1,3,5,7]   
     
    random.shuffle(ENTERANCE_positions)
    ENTERANCE_position=ENTERANCE_positions[0]
    ENTERANCE_positions.remove(ENTERANCE_position)
    positions.remove(ENTERANCE_position)
    rooms[int(ENTERANCE_position/3)][int(ENTERANCE_position%3)]=create_single_room_definition(flat_width,flat_depth,"ENTERANCE","HALL","#0000FF",2,3,int(ENTERANCE_position/3),int(ENTERANCE_position%3),MAX_I,MAX_J)
    
    CENTRE_POSITION=4
    positions.remove(CENTRE_POSITION)
    rooms[int(CENTRE_POSITION/3)][int(CENTRE_POSITION%3)]=create_single_room_definition(flat_width,flat_depth,"CENTRE","HALL","#0000FF",2,2,int(CENTRE_POSITION/3),int(CENTRE_POSITION%3),MAX_I,MAX_J)
     
    random.shuffle(ENTERANCE_positions)
    BATH_ROOM_POSITION=ENTERANCE_positions[0]
    ENTERANCE_positions.remove(BATH_ROOM_POSITION)
    positions.remove(BATH_ROOM_POSITION)
    rooms[int(BATH_ROOM_POSITION/3)][int(BATH_ROOM_POSITION%3)]=create_single_room_definition(flat_width,flat_depth,"BATH_ROOM_1","BATH_ROOM","#00FF00",2,6,int(BATH_ROOM_POSITION/3),int(BATH_ROOM_POSITION%3),MAX_I,MAX_J)
            
    random.shuffle(ENTERANCE_positions)
    BATH_ROOM_POSITION=ENTERANCE_positions[0]
    ENTERANCE_positions.remove(BATH_ROOM_POSITION)
    positions.remove(BATH_ROOM_POSITION)
    rooms[int(BATH_ROOM_POSITION/3)][int(BATH_ROOM_POSITION%3)]=create_single_room_definition(flat_width,flat_depth,"BATH_ROOM_2","BATH_ROOM","#FFFF00",2,6,int(BATH_ROOM_POSITION/3),int(BATH_ROOM_POSITION%3),MAX_I,MAX_J)
            
    KITCHEN_ROOM_POSITION=ENTERANCE_positions[0]
    ENTERANCE_positions.remove(KITCHEN_ROOM_POSITION)
    positions.remove(KITCHEN_ROOM_POSITION)
    rooms[int(KITCHEN_ROOM_POSITION/3)][int(KITCHEN_ROOM_POSITION%3)]=create_single_room_definition(flat_width,flat_depth,"KITCHEN","KITCHEN","#00F0A0",2,6,int(KITCHEN_ROOM_POSITION/3),int(KITCHEN_ROOM_POSITION%3),MAX_I,MAX_J)
            
    rooms[int(positions[0]/3)][int(positions[0]%3)]=create_single_room_definition(flat_width,flat_depth,"LIVING_ROOM_1","LIVING_ROOM","#660066",3,6,int(positions[0]/3),int(positions[0]%3),MAX_I,MAX_J)
    rooms[int(positions[1]/3)][int(positions[1]%3)]=create_single_room_definition(flat_width,flat_depth,"BED_ROOM_1","BED_ROOM","#006600",3,6,int(positions[1]/3),int(positions[1]%3),MAX_I,MAX_J)
    rooms[int(positions[2]/3)][int(positions[2]%3)]=create_single_room_definition(flat_width,flat_depth,"BED_ROOM_2","BED_ROOM","#999900",3,6,int(positions[2]/3),int(positions[2]%3),MAX_I,MAX_J)
    rooms[int(positions[3]/3)][int(positions[3]%3)]=create_single_room_definition(flat_width,flat_depth,"BED_ROOM_3","BED_ROOM","#FF00FF",3,6,int(positions[3]/3),int(positions[3]%3),MAX_I,MAX_J)
    
    return rooms

def grow_rooms(floor_plan):

    for i in range(len(floor_plan['rooms'])):
      for j in range(len(floor_plan['rooms'][i])):
        room=floor_plan["rooms"][i][j]
        random_int=random.randint(0, 3)
        random_2=random.randint(0, 10)
        random_3=random.randint(0, 1)
        if random_2 != 0 and room["type"] == "BATH_ROOM" :
           continue
        if random_3 != 0 and  room["type"] == "KITCHEN" :
           continue  
        if random_int == 0 and room["coordinates"]["x"]["start"]["growable"]     :  room["coordinates"]["x"]["start"]["value"]-=0.1
        if random_int == 1 and room["coordinates"]["x"]["end"]["growable"]       :  room["coordinates"]["x"]["end"]["value"]+=0.1
        if random_int == 2 and room["coordinates"]["y"]["start"]["growable"]     :  room["coordinates"]["y"]["start"]["value"]-=0.1
        if random_int == 3 and room["coordinates"]["y"]["end"]["growable"]       :  room["coordinates"]["y"]["end"]["value"]+=0.1
           
    
def generate_random_floor_plan(flat_width,flat_depth,flat_height,instance_id):
    floor_plan={}
    floor_plan["dimensions"]={}
    floor_plan["dimensions"]["width"]=flat_width
    floor_plan["dimensions"]["depth"]=flat_depth
    floor_plan["dimensions"]["height"]=flat_height
    floor_plan["id"]=instance_id
    floor_plan["rooms"]=create_random_room_definition(flat_width,flat_depth)
    return floor_plan



def collide_axis_bound(i1,j1,i2,j2,axis,bound,rooms):
    ##     i1.axis.bound is between i2.axis.start and i2.axis.end
    f1=(rooms[i1][j1]["coordinates"][axis][bound]["value"] >= rooms[i2][j2]["coordinates"][axis]["start"]["value"]-0.2)
    f2=(rooms[i1][j1]["coordinates"][axis][bound]["value"] <= rooms[i2][j2]["coordinates"][axis]["end"]["value"]+0.2)
    #if f1 and f2 : print(f"{i1}.{j1}.{axis}.{bound} ({rooms[i1][j1]['coordinates'][axis][bound]['value']}) is between {i2}.{j2}.{axis}.start ({rooms[i2][j2]['coordinates'][axis]['start']['value']}) and  {i2}.{j2}.{axis}.end ({rooms[i2][j2]['coordinates'][axis]['end']['value']})") 
    return (f1 and f2)
     
def collide_axis(i1,j1,i2,j2,axis,rooms):
    return collide_axis_bound(i1,j1,i2,j2,axis,"start",rooms) or collide_axis_bound(i1,j1,i2,j2,axis,"end",rooms) 

def collides(i1,j1,i2,j2,rooms):
    return collide_axis(i1,j1,i2,j2,"x",rooms) and collide_axis(i1,j1,i2,j2,"y",rooms)


def check_collision(rooms):

    for i1 in range(len(rooms)) :
      for j1 in range(len(rooms[i1])):
         for i2 in range(len(rooms)) :
            for j2 in range(len(rooms[i2])) :
              if i1==i2 and j1==j2 :
               continue
              else:
               if collides(i1,j1,i2,j2,rooms):
                 #print(f"{i1}.{j1} - {i2}.{j2} collides")
                 axis="x"
                 if collide_axis(i1,j1,i2,j2,axis,rooms) :
                   if i1 < i2 and collide_axis(i1,j1,i2,j2,axis,rooms) :
                      rooms[i1][j1]["coordinates"][axis]["end"]["growable"]=False 
                      rooms[i2][j2]["coordinates"][axis]["start"]["growable"]=False 
                   elif i1>i2: 
                      rooms[i1][j1]["coordinates"][axis]["start"]["growable"]=False 
                      rooms[i2][j2]["coordinates"][axis]["end"]["growable"]=False 
                 axis="y"
                 if collide_axis(i1,j1,i2,j2,axis,rooms) :
                   if j1 < j2 and collide_axis(i1,j1,i2,j2,axis,rooms) :
                      rooms[i1][j1]["coordinates"][axis]["end"]["growable"]=False 
                      rooms[i2][j2]["coordinates"][axis]["start"]["growable"]=False 
                   elif j1>j2: 
                      rooms[i1][j1]["coordinates"][axis]["start"]["growable"]=False 
                      rooms[i2][j2]["coordinates"][axis]["end"]["growable"]=False 

    for i in range(len(rooms)) :
      for j in range(len(rooms[i])):
        for xy in ["x","y"]:
            if abs(rooms[i][j]["coordinates"][xy]["start"]["value"]- rooms[i][j]["coordinates"][xy]["end"]["value"])>=rooms[i][j]["max_len"]: 
                  rooms[i][j]["coordinates"][xy]["start"]["growable"]=False 
                  rooms[i][j]["coordinates"][xy]["end"]["growable"]=False 


    noSpaceLeftToGrow=True
    for i in range(len(rooms)) :
      for j in range(len(rooms[i])):
        room=rooms[i][j]
        for axis in ["x","y"]:
          for se in ["start","end"]:
            if room["coordinates"][axis][se]["growable"] :
               noSpaceLeftToGrow=False
        if not noSpaceLeftToGrow:
           break

    return not noSpaceLeftToGrow



def close_axis_bound(i1,j1,i2,j2,axis,bound,rooms):
    ##     i1.axis.bound is between i2.axis.start and i2.axis.end
    DOOR_WIDTH=1.5
    f1=(rooms[i1][j1]["coordinates"][axis][bound]["value"] >= rooms[i2][j2]["coordinates"][axis]["start"]["value"]-DOOR_WIDTH)
    f2=(rooms[i1][j1]["coordinates"][axis][bound]["value"] <= rooms[i2][j2]["coordinates"][axis]["end"]["value"]+DOOR_WIDTH)
    #if f1 and f2 : print(f"{i1}.{j1}.{axis}.{bound} ({rooms[i1][j1]['coordinates'][axis][bound]['value']}) is between {i2}.{j2}.{axis}.start ({rooms[i2][j2]['coordinates'][axis]['start']['value']}) and  {i2}.{j2}.{axis}.end ({rooms[i2][j2]['coordinates'][axis]['end']['value']})") 
    return (f1 and f2)
     


def adjust_for_doors(rooms):

    # 0 1 2
    # 3 4 5
    # 6 7 8
    
    # 0.0 0.1 0.2
    # 1.0 1.1 1.2
    # 2.0 2.1 2.2
    
    noDoorRooms=[]
    

    for i1 in range(len(rooms)) :
      for j1 in range(len(rooms[i1])):
          if rooms[i1][j1]["type"] == "HALL" :
             continue   
          noDoor=True
          if i1-1 >= 0 and (rooms[i1-1][j1]["type"] == "HALL" or not close_axis_bound(i1-1,j1,i1,j1,"x","end",rooms) ):
               noDoor=False
          if i1+1 <= len(rooms)-1 and  (rooms[i1+1][j1]["type"] == "HALL" or not close_axis_bound(i1,j1,i1+1,j1,"x","end",rooms)):
             noDoor=False
          if j1-1 >= 0 and  (rooms[i1][j1-1]["type"] == "HALL" or not close_axis_bound(i1,j1-1,i1,j1,"y","end",rooms)):
             noDoor=False
          if j1+1 <= len(rooms[i1])-1 and  (rooms[i1][j1+1]["type"] == "HALL" or not close_axis_bound(i1,j1,i1,j1+1,"y","end",rooms)):
             noDoor=False
          if noDoor :
             noDoorRooms.append(rooms[i1][j1])
      
    #for room in noDoorRooms:
    #    print(f"NO DOOR ROOM : {room['region']['x']}.{room['region']['y']}") 
    
    if len(noDoorRooms) == 0 :
        print("All rooms have doors :) ")
    elif len(noDoorRooms) > 2 :
       print("Cant do anything for noDoorRoom > 2, it is recomended to delete this room configuration ... ")
    elif len(noDoorRooms) == 2 :
       room0=noDoorRooms[0]
       room1=noDoorRooms[1]
       orientation="y"
       room0x=0
       room0y=0
       room1x=0
       room1y=0
       
       if room0['region']['x']==room1['region']['x'] :
         orientation="y"
         if room0['region']['x'] == 0 :
          #if room0["coordinates"]["x"]["end"]["value"] - rooms[room0['region']['x']][1]["coordinates"]["x"]["end"]["value"] < 1 and room1["coordinates"]["x"]["end"]["value"] - rooms[room0['region']['x']][1]["coordinates"]["x"]["end"]["value"] < 1:
          rooms[room0['region']['x']][1]["coordinates"]["x"]["end"]["value"]=rooms[room0['region']['x']][1]["coordinates"]["x"]["end"]["value"]-1
          #room0x=rooms[room0['region']['x']][1]["coordinates"]["x"]["end"]["value"]+1/2
          room0x=room0["coordinates"]["x"]["end"]["value"]-1/2-0.1
          room0y=room0["coordinates"]["y"]["end"]["value"]
          #room1x=room0x
          room1x=room1["coordinates"]["x"]["end"]["value"]-1/2-0.1
          room1y=room1["coordinates"]["y"]["start"]["value"]
         else:
          #if room0["coordinates"]["x"]["start"]["value"] - rooms[room0['region']['x']][1]["coordinates"]["x"]["start"]["value"] <1 and room1["coordinates"]["x"]["start"]["value"] - rooms[room0['region']['x']][1]["coordinates"]["x"]["start"]["value"] <1 : 
          rooms[room0['region']['x']][1]["coordinates"]["x"]["start"]["value"]=rooms[room0['region']['x']][1]["coordinates"]["x"]["start"]["value"]+1
          #room0x=rooms[room0['region']['x']][1]["coordinates"]["x"]["start"]["value"]-1/2
          room0x=room0["coordinates"]["x"]["start"]["value"]+1/2+0.1
          room0y=room0["coordinates"]["y"]["end"]["value"]
          #room1x=room0x
          room1x=room1["coordinates"]["x"]["start"]["value"]+1/2+0.1
          room1y=room1["coordinates"]["y"]["start"]["value"]
       else:    
         orientation="x" 
         if room0['region']['y'] == 0 :
          rooms[1][room0['region']['y']]["coordinates"]["y"]["end"]["value"]=rooms[1][room0['region']['y']]["coordinates"]["y"]["end"]["value"]-1
          room0x=room0["coordinates"]["x"]["end"]["value"]
          #room0y=rooms[1][room0['region']['y']]["coordinates"]["y"]["end"]["value"]+1/2
          room0y=room0["coordinates"]["y"]["end"]["value"]-1/2-0.1
          room1x=room1["coordinates"]["x"]["start"]["value"]
          #room1y=room0y
          room1y=room1["coordinates"]["y"]["end"]["value"]-1/2-0.1
         else:
          rooms[1][room0['region']['y']]["coordinates"]["y"]["start"]["value"]=rooms[1][room0['region']['y']]["coordinates"]["y"]["start"]["value"]+1
          room0x=room0["coordinates"]["x"]["end"]["value"]
          #room0y=rooms[1][room0['region']['y']]["coordinates"]["y"]["start"]["value"]-1/2
          room0y=room0["coordinates"]["y"]["start"]["value"]+1/2+0.1
          room1x=room1["coordinates"]["x"]["start"]["value"]
          #room1y=room0y
          room1y=room1["coordinates"]["y"]["start"]["value"]+1/2+0.1


       room0["door"]={}
       room0["door"]["x"]=room0x
       room0["door"]["y"]=room0y
       room0["door"]["orientation"]=orientation
       

       room1["door"]={}
       room1["door"]["x"]=room1x
       room1["door"]["y"]=room1y
       room1["door"]["orientation"]=orientation
               
          
    else:
       ## len(noDoorRooms) == 1  case
       room0=noDoorRooms[0]
       room0x=room0["coordinates"]["x"]["start"]["value"]
       room0y=room0["coordinates"]["y"]["start"]["value"]
       room0["door"]={}
       room0["door"]["x"]=room0x
       room0["door"]["y"]=room0y
       room0["door"]["orientation"]="x"

               




    ## locate door center      
    for i1 in range(len(rooms)) :
      for j1 in range(len(rooms[i1])):
          if rooms[i1][j1]["type"] == "HALL" :
             continue   
          if "door" not in rooms[i1][j1]:
              rooms[i1][j1]["door"]={}
          if i1-1 >= 0 and (rooms[i1-1][j1]["type"] == "HALL" or not close_axis_bound(i1-1,j1,i1,j1,"x","end",rooms) ):
               rooms[i1][j1]["door"]["x"]=rooms[i1][j1]["coordinates"]["x"]["start"]["value"]
               rooms[i1][j1]["door"]["y"]=(rooms[i1][j1]["coordinates"]["y"]["start"]["value"]+rooms[i1][j1]["coordinates"]["y"]["end"]["value"])/2
               rooms[i1][j1]["door"]["orientation"]="x"
          if i1+1 <= len(rooms)-1 and  (rooms[i1+1][j1]["type"] == "HALL" or not close_axis_bound(i1,j1,i1+1,j1,"x","end",rooms)):
               rooms[i1][j1]["door"]["x"]=rooms[i1][j1]["coordinates"]["x"]["end"]["value"]
               rooms[i1][j1]["door"]["y"]=(rooms[i1][j1]["coordinates"]["y"]["start"]["value"]+rooms[i1][j1]["coordinates"]["y"]["end"]["value"])/2
               rooms[i1][j1]["door"]["orientation"]="x"
          if j1-1 >= 0 and  (rooms[i1][j1-1]["type"] == "HALL" or not close_axis_bound(i1,j1-1,i1,j1,"y","end",rooms)):
               rooms[i1][j1]["door"]["x"]=(rooms[i1][j1]["coordinates"]["x"]["start"]["value"]+rooms[i1][j1]["coordinates"]["x"]["end"]["value"])/2
               rooms[i1][j1]["door"]["y"]=rooms[i1][j1]["coordinates"]["y"]["start"]["value"]
               rooms[i1][j1]["door"]["orientation"]="y"
          if j1+1 <= len(rooms[i1])-1 and  (rooms[i1][j1+1]["type"] == "HALL" or not close_axis_bound(i1,j1,i1,j1+1,"y","end",rooms)):
               rooms[i1][j1]["door"]["x"]=(rooms[i1][j1]["coordinates"]["x"]["start"]["value"]+rooms[i1][j1]["coordinates"]["x"]["end"]["value"])/2
               rooms[i1][j1]["door"]["y"]=rooms[i1][j1]["coordinates"]["y"]["end"]["value"]
               rooms[i1][j1]["door"]["orientation"]="y"
          
          


    
def generate_floor_plans(width,depth,height,available_furnitures,NUMBER_OF_SAMPLES,DATA_DIR):
    if depth<10 or width<10:
       print("Sorry, you cannot define any dimension less then 10 meters ..")
       return None
       
    floor_plans=[]        
    NUMBER_OF_ITERATIONS=500
    for i in range(int(NUMBER_OF_SAMPLES)):
        #print(f"Generating floor plan id : {i} with {width}x{depth}x{height}")
        floor_plan=generate_random_floor_plan(width,depth,height,str(i))
        for i in range(NUMBER_OF_ITERATIONS):
           if not check_collision(floor_plan["rooms"]):
              break
           grow_rooms(floor_plan)
        adjust_for_doors(floor_plan["rooms"])

        decorate_rooms(floor_plan,available_furnitures)
        
        # sonradan verinin istatistigini toplamak istersem, odalarin boyutlari ile ilgili cesitlilik dagilimi vs. icin
        save_floor_plan(floor_plan,DATA_DIR)
#        floor_plan_reloaded=load_floor_plan(DATA_DIR,floor_plan['id'])
#        plot_floor_plan(floor_plan_reloaded,DATA_DIR)
        plot_floor_plan(floor_plan,DATA_DIR)
    
        floor_plans.append(floor_plan)
    
    return floor_plans
    

def plot_floor_plan(floor_plan,DATA_DIR):

    main_im = Image.new('RGB', (floor_plan["dimensions"]["width"]*100, floor_plan["dimensions"]["depth"]*100))
    ImageDraw.Draw(main_im).rectangle([(0,0),(floor_plan["dimensions"]["width"]*100,floor_plan["dimensions"]["depth"]*100)],fill="white")
    fontsize=20
    rooms=floor_plan["rooms"]
    
    for i in range(len(rooms)) :
      for j in range(len(rooms[i])):
        room=rooms[i][j]
        ImageDraw.Draw(main_im).rectangle([(room["coordinates"]["x"]["start"]["value"]*100,room["coordinates"]["y"]["start"]["value"]*100),(room["coordinates"]["x"]["end"]["value"]*100,room["coordinates"]["y"]["end"]["value"]*100)],fill=room["color"])
        ImageDraw.Draw(main_im).text((room["coordinates"]["x"]["start"]["value"]*100+10,room["coordinates"]["y"]["start"]["value"]*100+10),room["name"]+"\n"+str(i)+"."+str(j),fill="black",font=ImageFont.truetype("FreeSerifBold.ttf", fontsize))
#        if "door" in room and "orientation" in room["door"] :
#           if room["door"]["orientation"] == "x" :
#              ImageDraw.Draw(main_im).rectangle([(room["door"]["x"]*100,(room["door"]["y"]+1/2)*100),((room["door"]["x"]+0.1)*100,(room["door"]["y"]-1/2)*100)],fill="#000000")
#           else:
#              ImageDraw.Draw(main_im).rectangle([((room["door"]["x"]+1/2)*100,room["door"]["y"]*100),((room["door"]["x"]-1/2)*100,(room["door"]["y"]+0.1)*100)],fill="#000000")


#    main_im.save(DATA_DIR+f"/floor_plan-{floor_plan['id']}.flat.png")
    
    
    
    main_im = Image.new('RGB', (floor_plan["dimensions"]["width"]*100, floor_plan["dimensions"]["depth"]*100))
    ImageDraw.Draw(main_im).rectangle([(0,0),(floor_plan["dimensions"]["width"]*100,floor_plan["dimensions"]["depth"]*100)],fill="white")
    fontsize=20
         
    for i in range(len(rooms)) :
      for j in range(len(rooms[i])):
        room=rooms[i][j]
        ImageDraw.Draw(main_im).rectangle([(room["coordinates"]["x"]["start"]["value"]*100,room["coordinates"]["y"]["start"]["value"]*100),(room["coordinates"]["x"]["end"]["value"]*100,room["coordinates"]["y"]["end"]["value"]*100)],fill=room["color"])
        ImageDraw.Draw(main_im).text((room["coordinates"]["x"]["start"]["value"]*100+10,room["coordinates"]["y"]["start"]["value"]*100+10),room["name"]+"\n"+str(i)+"."+str(j),fill="black",font=ImageFont.truetype("FreeSerifBold.ttf", fontsize))
#        if "door" in room and "orientation" in room["door"] :
#           if room["door"]["orientation"] == "x" :
#              ImageDraw.Draw(main_im).rectangle([(room["door"]["x"]*100,(room["door"]["y"]+1/2)*100),((room["door"]["x"]+0.1)*100,(room["door"]["y"]-1/2)*100)],fill="#000000")
#           else:
#              ImageDraw.Draw(main_im).rectangle([((room["door"]["x"]+1/2)*100,room["door"]["y"]*100),((room["door"]["x"]-1/2)*100,(room["door"]["y"]+0.1)*100)],fill="#000000")
        
        
        #I=int(abs(room["coordinates"]["x"]["end"]["value"]-room["coordinates"]["x"]["start"]["value"])*10)
        #J=int(abs(room["coordinates"]["y"]["end"]["value"]-room["coordinates"]["y"]["start"]["value"])*10)
  
             
                    
        if "furnitures" in room: ## HALL does not have furniture
        
        
        
 
         for furniture in room["furnitures"]:
            i1=furniture["start_region"]["i"]
            j1=furniture["start_region"]["j"]
            i2=furniture["end_region"]["i"]
            j2=furniture["end_region"]["j"]


            x1=room["coordinates"]["x"]["start"]["value"]*100+i1*10
            x2=room["coordinates"]["x"]["start"]["value"]*100+i2*10
            y1=room["coordinates"]["y"]["start"]["value"]*100+j1*10
            y2=room["coordinates"]["y"]["start"]["value"]*100+j2*10
            
            
            #print(f"Drawing {furniture['name']} on {room['name']} with dimensions : ###  {x1} # {i1} ## {y1} # {j1}  ### {x2} # {i2} ## {y2} # {j2} ")  
            ImageDraw.Draw(main_im, "RGBA").rectangle([(x1,y1),(x2,y2)],fill=furniture["color"]+"AA")
            ImageDraw.Draw(main_im).text((x1+10,y1+10), furniture["name"],fill="black", font=ImageFont.truetype("FreeSerifBold.ttf",fontsize))
            
            #break

         '''
         if "matrix" in room:
          #print("PLOTTING IMAGE OF ROOM")
          #print(room["matrix"])
          for i1 in range(I):
            for j1 in range(J):
              if "matrix" in room and int(room["matrix"][i1][j1]) == 2:
                 x1=room["coordinates"]["x"]["start"]["value"]*100+i1*10
                 y1=room["coordinates"]["y"]["start"]["value"]*100+j1*10
                 print(f"DOOR i1={i1} j1={j1} x1={x1} y1={y1}")
                 ImageDraw.Draw(main_im,).rectangle([(x1,y1),(x1+10,y1+10)],fill="#FF0000")  
              if "matrix" in room and int(room["matrix"][i1][j1]) == 3:
                 x1=room["coordinates"]["x"]["start"]["value"]*100+i1*10
                 y1=room["coordinates"]["y"]["start"]["value"]*100+j1*10
                 ImageDraw.Draw(main_im,).rectangle([(x1,y1),(x1+10,y1+10)],fill="#000000")  
         '''

            
    main_im.save(DATA_DIR+f"/floor_plan-{floor_plan['id']}.furnitures.png")
       


def save_floor_plan(floor_plan,DATA_DIR):
    with open(DATA_DIR+f"/floor_plan-{floor_plan['id']}.pickle", 'wb') as f:
        pickle.dump(floor_plan, f, protocol=2)

 
def load_floor_plan(DATA_DIR,floor_plan_id):
    with open(DATA_DIR+f"/floor_plan-{floor_plan_id}.pickle", 'rb') as f:
       floor_plan = pickle.load(f)
    return floor_plan

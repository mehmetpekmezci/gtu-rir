#!/usr/bin/env python


import numpy as np



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

     
          ## GTU-RIR DATA COORDS ##           
    ###################################       
    # 0.0          0.1            0.2 #       
    #                                 #       
    #                                 #       
    # 1.0          1.1            1.2 #       
    #                                 #       
    #                                 #       
    # 2.0          2.1            2.2 #       
    ###################################       

    #                Z
    #                |
    #                |
    #                |_____Y                        
    #               /                         
    #              /                          
    #            / X                   
    #                                         

    
    
    
def get_gtu_room_data():

 GTU_ROOM={}
#GTU_ROOM["207"]={}
#GTU_ROOM["208"]={}
#GTU_ROOM["conferrence01"]={}
#GTU_ROOM["sport01"]={}
#GTU_ROOM["sport02"]={}
#GTU_ROOM["z02"]={}
#GTU_ROOM["z04"]={}
#GTU_ROOM["z06"]={}
#GTU_ROOM["z10"]={}
#GTU_ROOM["z11"]={}
#GTU_ROOM["z23"]={}



# #########
# 0,0,0
#
#
#


### 0,0,0 = KARSI DUVARIN EN SOLU
 FURNITURE_TYPE={}
 FURNITURE_TYPE["CHAIR"]={}
 FURNITURE_TYPE["CHAIR"]["WIDTH"]=53/100
 FURNITURE_TYPE["CHAIR"]["DEPTH"]=(9+59)/100
 FURNITURE_TYPE["CHAIR"]["HEIGHT"]=77/100
 FURNITURE_TYPE["CHAIR"]["OTURAK"]={}
 FURNITURE_TYPE["CHAIR"]["OTURAK"]["WIDTH"]=36/100
 FURNITURE_TYPE["CHAIR"]["OTURAK"]["DEPTH"]=39/100
 FURNITURE_TYPE["CHAIR"]["OTURAK"]["HEIGHT"]=2/100
 FURNITURE_TYPE["CHAIR"]["OTURAK"]["HEIGHT_FROM_GROUND"]=41/100
 FURNITURE_TYPE["CHAIR"]["ARKA"]={}
 FURNITURE_TYPE["CHAIR"]["ARKA"]["WIDTH"]=41/100
 FURNITURE_TYPE["CHAIR"]["ARKA"]["DEPTH"]=3/100
 FURNITURE_TYPE["CHAIR"]["ARKA"]["HEIGHT"]=34/100
 FURNITURE_TYPE["CHAIR"]["ARKA"]["HEIGHT_FROM_GROUND"]=43/100
 FURNITURE_TYPE["CHAIR"]["TABLA"]={}
 FURNITURE_TYPE["CHAIR"]["TABLA"]["WIDTH"]=27/100
 FURNITURE_TYPE["CHAIR"]["TABLA"]["WIDTH_FROM_BACK"]=11/100
 FURNITURE_TYPE["CHAIR"]["TABLA"]["DEPTH"]=59/100
 FURNITURE_TYPE["CHAIR"]["TABLA"]["DEPTH_FROM_BACK"]=9/100
 FURNITURE_TYPE["CHAIR"]["TABLA"]["HEIGHT"]=2/100
 FURNITURE_TYPE["CHAIR"]["TABLA"]["HEIGHT_FROM_GROUND"]=68/100


 FURNITURE_TYPE["KONSOLE1"]={}
 FURNITURE_TYPE["KONSOLE1"]["WIDTH"]=542.0/100
 FURNITURE_TYPE["KONSOLE1"]["DEPTH"]=30/100
 FURNITURE_TYPE["KONSOLE1"]["HEIGHT"]=77 /100


 FURNITURE_TYPE["KONSOLE2"]={}
 FURNITURE_TYPE["KONSOLE2"]["WIDTH"]=31.5/100
 FURNITURE_TYPE["KONSOLE2"]["DEPTH"]=977.0/100
 FURNITURE_TYPE["KONSOLE2"]["HEIGHT"]=77 /100

 FURNITURE_TYPE["TABLE"]={}
 FURNITURE_TYPE["TABLE"]["WIDTH"]=120/100
 FURNITURE_TYPE["TABLE"]["DEPTH"]=60/100
 FURNITURE_TYPE["TABLE"]["HEIGHT"]=2/100
 FURNITURE_TYPE["TABLE"]["HEIGHT_FROM_GROUND"]=77/100


### 0,0,0 = KAPDAN GIRNCE KARSI DUVARIN EN SOLU

### ROOM DIMENSIONS
 GTU_ROOM["208"]={}  
 GTU_ROOM["208"]["DEPTH"]=977/100  ## X
 GTU_ROOM["208"]["WIDTH"]=562.0/100   ## Y
 GTU_ROOM["208"]["HEIGHT"]=283.0/100  ## Z

 GTU_ROOM["z10"]={}  
 GTU_ROOM["z10"]["DEPTH"]=854.0/100
 GTU_ROOM["z10"]["WIDTH"]=1184.0/100
 GTU_ROOM["z10"]["HEIGHT"]=325.0/100


### ROOM FURNITURES
 GTU_ROOM["208"]["FURNITURE_ARRAY"]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][0]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][0]["TYPE"]="KONSOLE1"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][0]["ORIENTATION"]=0
 GTU_ROOM["208"]["FURNITURE_ARRAY"][0]["COUNT"]=1
 GTU_ROOM["208"]["FURNITURE_ARRAY"][0]["X"]= 0
 GTU_ROOM["208"]["FURNITURE_ARRAY"][0]["Y"]= 0

 GTU_ROOM["208"]["FURNITURE_ARRAY"][1]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][1]["TYPE"]="KONSOLE2"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][1]["ORIENTATION"]=0
 GTU_ROOM["208"]["FURNITURE_ARRAY"][1]["COUNT"]=1
 GTU_ROOM["208"]["FURNITURE_ARRAY"][1]["X"]=0
 GTU_ROOM["208"]["FURNITURE_ARRAY"][1]["Y"]=GTU_ROOM["208"]["WIDTH"]-FURNITURE_TYPE["KONSOLE2"]["WIDTH"]


 GTU_ROOM["208"]["FURNITURE_ARRAY"][2]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][2]["TYPE"]="CHAIR"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][2]["ORIENTATION"]=np.pi/2
 GTU_ROOM["208"]["FURNITURE_ARRAY"][2]["COUNT"]=17
 GTU_ROOM["208"]["FURNITURE_ARRAY"][2]["X"]=30/100
 GTU_ROOM["208"]["FURNITURE_ARRAY"][2]["Y"]=0
#En son sırada 17 CHAIR var, 2lik de boş yer var sanırım

 GTU_ROOM["208"]["FURNITURE_ARRAY"][3]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][3]["TYPE"]="CHAIR"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][3]["ORIENTATION"]=np.pi/2
 GTU_ROOM["208"]["FURNITURE_ARRAY"][3]["COUNT"]=14
 GTU_ROOM["208"]["FURNITURE_ARRAY"][3]["X"]=30/100
 GTU_ROOM["208"]["FURNITURE_ARRAY"][3]["Y"]=FURNITURE_TYPE["CHAIR"]["DEPTH"]/4*2.5+FURNITURE_TYPE["CHAIR"]["DEPTH"]
#sırada 11. beyaz CHAIR var, toplamda 14

 GTU_ROOM["208"]["FURNITURE_ARRAY"][4]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][4]["TYPE"]="CHAIR"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][4]["ORIENTATION"]=np.pi/2
 GTU_ROOM["208"]["FURNITURE_ARRAY"][4]["COUNT"]=8
 GTU_ROOM["208"]["FURNITURE_ARRAY"][4]["X"]=30/100
 GTU_ROOM["208"]["FURNITURE_ARRAY"][4]["Y"]=2*(FURNITURE_TYPE["CHAIR"]["DEPTH"]/4*2.5+FURNITURE_TYPE["CHAIR"]["DEPTH"])
#sırada 8 

 GTU_ROOM["208"]["FURNITURE_ARRAY"][5]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][5]["TYPE"]="CHAIR"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][5]["ORIENTATION"]=np.pi/2
 GTU_ROOM["208"]["FURNITURE_ARRAY"][5]["COUNT"]=8
 GTU_ROOM["208"]["FURNITURE_ARRAY"][5]["X"]=30/100+9*FURNITURE_TYPE["CHAIR"]["WIDTH"]
 GTU_ROOM["208"]["FURNITURE_ARRAY"][5]["Y"]=2*(FURNITURE_TYPE["CHAIR"]["DEPTH"]/4*2.5+FURNITURE_TYPE["CHAIR"]["DEPTH"])
#sırada 8 

 GTU_ROOM["208"]["FURNITURE_ARRAY"][6]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][6]["TYPE"]="CHAIR"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][6]["ORIENTATION"]=np.pi/2
 GTU_ROOM["208"]["FURNITURE_ARRAY"][6]["COUNT"]=3
 GTU_ROOM["208"]["FURNITURE_ARRAY"][6]["X"]=30/100
 GTU_ROOM["208"]["FURNITURE_ARRAY"][6]["Y"]=3*(FURNITURE_TYPE["CHAIR"]["DEPTH"]/4*2.5+FURNITURE_TYPE["CHAIR"]["DEPTH"])
#sırada 8 


 GTU_ROOM["208"]["FURNITURE_ARRAY"][7]={}
 GTU_ROOM["208"]["FURNITURE_ARRAY"][7]["TYPE"]="TABLE"
 GTU_ROOM["208"]["FURNITURE_ARRAY"][7]["ORIENTATION"]=-np.pi/2
 GTU_ROOM["208"]["FURNITURE_ARRAY"][7]["COUNT"]=1
 GTU_ROOM["208"]["FURNITURE_ARRAY"][7]["X"]=400/100
 GTU_ROOM["208"]["FURNITURE_ARRAY"][7]["Y"]=3*(FURNITURE_TYPE["CHAIR"]["DEPTH"]/4*2.5+FURNITURE_TYPE["CHAIR"]["DEPTH"])+FURNITURE_TYPE["CHAIR"]["DEPTH"]
#sırada 8 
 return FURNITURE_TYPE,GTU_ROOM



#GTU_ROOM["208"]["FURNITURE"][0]["TYPE"]="CHAIR"
#GTU_ROOM["208"]["FURNITURE"][0]["ORIENTATION"]="WIDTH=DEPTH" ## == DEPTH=WIDTH
#GTU_ROOM["208"]["FURNITURE"][0]["COORDS"]=[]










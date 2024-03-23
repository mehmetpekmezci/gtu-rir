#!/usr/bin/env python3
##
## IMPORTS
##
import importlib
csv         = importlib.import_module("csv")
sys         = importlib.import_module("sys")
np          = importlib.import_module("numpy")
os          = importlib.import_module("os")
json        = importlib.import_module("json")


def write_config(save_path, scene):
    with open(save_path, 'w') as outfile:
        json.dump(scene, outfile, sort_keys=True, indent=4)

if len(sys.argv) < 2 :
   print("Usage : python3 generate_sim_config_json_from_data_stats_csv.py <GWA_DATASET_DIR> <3D_FRONT_DIR>")
   exit(1)

GWA_DATASET_DIR=sys.argv[1]
E_3D_FRONT_DIR=sys.argv[2]


if not os.path.exists(GWA_DATASET_DIR+"/stats.csv") :
   print(GWA_DATASET_DIR+"/stats.csv not found !")
   exit(1)


if not os.path.exists(E_3D_FRONT_DIR) :
    print(E_3D_FRONT_DIR+" directory noot found !")
    exit(1)


gwa_dataset_stats=np.array(np.loadtxt(open(GWA_DATASET_DIR+"/stats.csv", 'r'), delimiter=",",dtype=str))

##print(gwa_dataset_stats.shape)

SCENEs={}

for i in range(gwa_dataset_stats.shape[0]):
    if gwa_dataset_stats[i][0].startswith('Path') :
        continue
    #print(gwa_dataset_stats[i][0]+","+gwa_dataset_stats[i][8]+","+gwa_dataset_stats[i][9])
    SCENE_CODE=gwa_dataset_stats[i][0].split('/')[0]
    WAV_FILE=gwa_dataset_stats[i][0].split('/')[1].replace('.wav','')
    SOURCE_NUMBER=int(WAV_FILE.split('_')[0].replace('L',''))
    RECEIVER_NUMBER=int(WAV_FILE.split('_')[1].replace('R',''))
    #print(f"SCENE_CODE={SCENE_CODE} WAV_FILE={WAV_FILE} SOURCE_NUMBER={SOURCE_NUMBER} RECEIVER_NUMBER={RECEIVER_NUMBER}")
    #print(list(map(list,gwa_dataset_stats[i][8].split())))
    SOURCE_POS_XYZ=gwa_dataset_stats[i][8].replace('[','').replace(']','').split()
    RECEIVER_POS_XYZ=gwa_dataset_stats[i][9].replace('[','').replace(']','').split()

    
    if not (SCENE_CODE in SCENEs) :
       SCENEs[SCENE_CODE]={}
       SCENEs[SCENE_CODE]["obj_path"]="house.obj"
       SCENEs[SCENE_CODE]["receivers"]=[]
       SCENEs[SCENE_CODE]["sources"]=[]

    SCENE=SCENEs[SCENE_CODE]

    RECEIVER={}
    RECEIVER["name"]="R"+str(RECEIVER_NUMBER)
    RECEIVER["xyz"]=RECEIVER_POS_XYZ

    SCENE["receivers"].append(RECEIVER)

    SOURCE={}
    SOURCE["name"]="S"+str(SOURCE_NUMBER)
    SOURCE["xyz"]=SOURCE_POS_XYZ

    source_exists=False;
    for source in SCENE["sources"]:
        if SOURCE["name"] == source["name"] :
           source_exists=True
           break

    if not source_exists :
       SCENE["sources"].append(SOURCE)



for SCENE_CODE in SCENEs:
    save_path = os.path.join(E_3D_FRONT_DIR, SCENE_CODE, "sim_config.json")
    print("writing to : "+save_path)
    write_config(save_path, SCENEs[SCENE_CODE])


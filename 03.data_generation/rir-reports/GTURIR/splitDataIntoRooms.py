#!/usr/bin/env python3
import importlib
math        = importlib.import_module("math")
logging     = importlib.import_module("logging")
urllib3     = importlib.import_module("urllib3")
tarfile     = importlib.import_module("tarfile")
csv         = importlib.import_module("csv")
glob        = importlib.import_module("glob")
sys         = importlib.import_module("sys")
os          = importlib.import_module("os")
argparse    = importlib.import_module("argparse")
np          = importlib.import_module("numpy")
librosa     = importlib.import_module("librosa")
pandas      = importlib.import_module("pandas")
time        = importlib.import_module("time")
random      = importlib.import_module("random")
datetime    = importlib.import_module("datetime")
#keras       = importlib.import_module("keras")
gc          = importlib.import_module("gc")
wave        = importlib.import_module("wave")
scipy       = importlib.import_module("scipy")
#cv2         = importlib.import_module("cv2") 
configparser= importlib.import_module("configparser")
subprocess  = importlib.import_module("subprocess")
pickle      = importlib.import_module("pickle")
matplotlib  = importlib.import_module("matplotlib")



gtu_rir_data_pickle_file=str(sys.argv[1]).strip()
rir_data=[]

if  os.path.exists(gtu_rir_data_pickle_file) :
         rir_data_file=open(gtu_rir_data_pickle_file,'rb')
         rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
else :
         print(f"{gtu_rir_data_pickle_file} not exists")
         exit(1)

                   

rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
                                "physicalSpeakerNo":6,"microphoneStandInitialCoordinateX":7,"microphoneStandInitialCoordinateY":8,"microphoneStandInitialCoordinateZ":9,"speakerStandInitialCoordinateX":10,
                                "speakerStandInitialCoordinateY":11,"speakerStandInitialCoordinateZ":12,"microphoneMotorPosition":13,"speakerMotorPosition":14,"temperatureAtMicrohponeStand":15,
                                "humidityAtMicrohponeStand":16,"temperatureAtMSpeakerStand":17,"humidityAtSpeakerStand":18,"tempHumTimestamp":19,"speakerRelativeCoordinateX":20,"speakerRelativeCoordinateY":21,
                                "speakerRelativeCoordinateZ":22,"microphoneStandAngle":23,"speakerStandAngle":24,"speakerAngleTheta":25,"speakerAnglePhi":26,"mic_RelativeCoordinateX":27,"mic_RelativeCoordinateY":28,
                                "mic_RelativeCoordinateZ":29,"mic_DirectionX":30,"mic_DirectionY":31,"mic_DirectionZ":32,"mic_Theta":33,"mic_Phi":34,"essFilePath":35,
                                "roomId":36,"configId":37,"micNo":38, ## THESE VALUES WILL BE PARSED FROM essFilePath
                                "roomWidth":39,"roomHeight":40,"roomDepth":41, ## THESE VALUES WILL BE RETREIVED FROM ROOM PREOPERTIES                              
                                "rt60":42, ## RT60 will be calculated                              
                                "rirData":43 ## will be loaded from wav file   
                              }



#closePointsSpkMicRoomId={} # {roomId, SpkCoord, MicCoord, rir[{ Data[], rt60, micDirection[X,Y,Z] }] }

list_of_room_ids=[ "room-207",  "room-208", "room-conferrence01", "room-sport01", "room-sport02", "room-z02", "room-z04" ,"room-z06" ,"room-z10", "room-z11" ,"room-z23" ]

print(list_of_room_ids)

for selectedRoomId in list_of_room_ids:
    room_data=[]
    for dataline in rir_data:
        roomId=dataline[int(rir_data_field_numbers['roomId'])] 
        if roomId == selectedRoomId :
            room_data.append(dataline)
    print(f"selectedRoomId={selectedRoomId} len(room_data)={len(room_data)}")
    rir_data_file=open(gtu_rir_data_pickle_file+"."+selectedRoomId,'wb')
    pickle.dump(room_data,rir_data_file)
    rir_data_file.close()


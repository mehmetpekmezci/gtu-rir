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
pickle      = importlib.import_module("pickle")


np.set_printoptions(threshold=sys.maxsize)

rir_data_file=str(sys.argv[1]).strip()

configs=[]
for i in range(11):
    configs.append(str(sys.argv[i+2]).strip().split("-"))

for config in configs :
    print(config)

rir_data=[]
with open(rir_data_file, 'rb') as f:
    rir_data=pickle.load(f)


print(f"pickle file {rir_data_file} is loaded ...")

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
                             

coordinates={}

CENT=100   

for i in range(len(rir_data)):
       dataline=rir_data[i] 
       roomId=dataline[int(rir_data_field_numbers['roomId'])] 
       speakerIterationNo=dataline[int(rir_data_field_numbers['speakerMotorIterationNo'])]
       microphoneIterationNo=dataline[int(rir_data_field_numbers['microphoneMotorIterationNo'])]
       physicalSpeakerNo=dataline[int(rir_data_field_numbers['physicalSpeakerNo'])] 
       micNo=dataline[int(rir_data_field_numbers['micNo'])] 

       found=False
 
       if i%1000==0:
           print(i)
           print(f"dataline : {roomId}-{speakerIterationNo}-{microphoneIterationNo}-{physicalSpeakerNo}-{micNo}")
       for config in configs :
           if roomId=="room-"+config[0] and microphoneIterationNo == config[1] and speakerIterationNo == config[2] and physicalSpeakerNo == config[3] and micNo == config[4]:
               print(f"found config : {config[0]}-{config[1]}-{config[2]}-{config[3]}-{config[4]}")
               found=True

       if not found  :
             continue

       print(roomId)
       record_name = f"{roomId}-micstep-{microphoneIterationNo}-spkstep-{speakerIterationNo}-spkno-{physicalSpeakerNo}-micno-{micNo}"
       record_file_name=record_name+".spk_mic.coordinates.txt"
       microphoneCoordinatesX=float(dataline[int(rir_data_field_numbers['microphoneStandInitialCoordinateX'])])/CENT +float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateX'])])/CENT # CM to M 
       microphoneCoordinatesY=float(dataline[int(rir_data_field_numbers['microphoneStandInitialCoordinateY'])])/CENT +float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateY'])])/CENT  # CM to M
       microphoneCoordinatesZ=float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateZ'])])/CENT
       speakerCoordinatesX=float(dataline[int(rir_data_field_numbers['speakerStandInitialCoordinateX'])])/CENT +float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateX'])])/CENT  # CM to M
       speakerCoordinatesY=float(dataline[int(rir_data_field_numbers['speakerStandInitialCoordinateY'])])/CENT +float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateY'])])/CENT  # CM to M
       speakerCoordinatesZ=float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateZ'])])/CENT 

          
          
       #print(roomWorkDir+"/"+wave_name+" filename="+essFilePath+"  rir_data rt60 : "+rt60)
        
       try:
         
         #real_rir=librosa.resample(rir_data[i][-1], orig_sr=44100, target_sr=sr) 
        
         f = open("data/properties/"+record_file_name, "w")
         f.write(f"mic_coords_xyz={microphoneCoordinatesX:.2f},{microphoneCoordinatesY:.2f},{microphoneCoordinatesZ:.2f}\n")
         f.write(f"spk_coords_xyz={speakerCoordinatesX:.2f},{speakerCoordinatesY:.2f},{speakerCoordinatesZ:.2f}\n")
         f.close()
       except:
           print("Exception: roomId="+roomId+", record_name="+record_name)
           traceback.print_exc()


 

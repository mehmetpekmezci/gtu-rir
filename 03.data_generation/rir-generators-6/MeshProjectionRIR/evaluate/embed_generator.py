import os
import json
import numpy as np
import random
import argparse
import pickle
import math
import sys



def permuteGANInput(x,y,z,roomDepth,roomWidth,roomHeight):
     return  x,roomWidth-y,z

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

roomProperties={}

rir_data_file_path=str(sys.argv[1]).strip()
main_dir=str(sys.argv[2]).strip()
dataset_name=str(sys.argv[3]).strip()
metadata_dirname=str(sys.argv[4]).strip()
metadata_dir = main_dir+"/"+metadata_dirname
mesh2ir_embeddings_dir=metadata_dir+"/Embeddings"

print(f"Loading pickle file : {rir_data_file_path}")
rir_data=[] 
if  os.path.exists( rir_data_file_path) :
         rir_data_file=open(rir_data_file_path,'rb')
         rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
else :
         print(rir_data_file_path+" does not exist")
         exit(1)


print(f"Generating embeddings for rir_data od length {len(rir_data)}")
total_number_of_records=0
mesh2irInputData={}
for dataline in rir_data:
              roomId=dataline[int(rir_data_field_numbers['roomId'])]
              configId=dataline[int(rir_data_field_numbers['configId'])]
              speakerMotorIterationNo=dataline[int(rir_data_field_numbers['speakerMotorIterationNo'])]
              microphoneMotorIterationNo=dataline[int(rir_data_field_numbers['microphoneMotorIterationNo'])]
              physicalSpeakerNo=dataline[int(rir_data_field_numbers['physicalSpeakerNo'])]
              micNo=dataline[int(rir_data_field_numbers['micNo'])]

              CENT=1 ## M / CM
              if dataset_name == "GTURIR" :
                  CENT=100
              roomDepth=float(dataline[int(rir_data_field_numbers['roomDepth'])])/CENT # CM to M
              roomWidth=float(dataline[int(rir_data_field_numbers['roomWidth'])])/CENT # CM to M
              roomHeight=float(dataline[int(rir_data_field_numbers['roomHeight'])])/CENT # CM to M
              microphoneCoordinatesX=float(dataline[int(rir_data_field_numbers['microphoneStandInitialCoordinateX'])])/CENT +float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateX'])])/CENT # CM to M
              microphoneCoordinatesY=float(dataline[int(rir_data_field_numbers['microphoneStandInitialCoordinateY'])])/CENT +float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateY'])])/CENT # CM to M
              microphoneCoordinatesZ=float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateZ'])])/CENT
              if dataset_name == "GTURIR" :
                 microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ=permuteGANInput(microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ,roomDepth,roomWidth,roomHeight)
              speakerCoordinatesX=float(dataline[int(rir_data_field_numbers['speakerStandInitialCoordinateX'])])/CENT +float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateX'])])/CENT # CM to M
              speakerCoordinatesY=float(dataline[int(rir_data_field_numbers['speakerStandInitialCoordinateY'])])/CENT +float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateY'])])/CENT # CM to M
              speakerCoordinatesZ=float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateZ'])])/CENT
              if dataset_name == "GTURIR" :
                 speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ=permuteGANInput(speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ,roomDepth,roomWidth,roomHeight)
              rt60=float(dataline[int(rir_data_field_numbers['rt60'])])
              #mesh2irInputDataLine=[roomId+"-freecad-mesh-Body.pickle",roomId+"-CONFIG_ID-"+configId,"SPEAKER_ITERATION-"+speakerMotorIterationNo+"-MICROPHONE_ITERATION-"+microphoneMotorIterationNo+"-PHYSICAL_SPEAKER_NO-"+physicalSpeakerNo+"-MICROPHONE_NO-"+micNo+".wav",[speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ],[microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ]]
              mesh2irInputDataLine=[roomId+"-freecad-mesh-Body.obj",configId,"SPEAKER_ITERATION-"+speakerMotorIterationNo+"-MICROPHONE_ITERATION-"+microphoneMotorIterationNo+"-PHYSICAL_SPEAKER_NO-"+physicalSpeakerNo+"-MICROPHONE_NO-"+micNo+".wav",[speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ],[microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ]]
              if not roomId  in roomProperties :
               roomProperties[roomId]=[roomWidth,roomHeight,roomDepth]
              if roomId not in mesh2irInputData :
                 mesh2irInputData[roomId]=[]
              mesh2irInputData[roomId].append(mesh2irInputDataLine)
              total_number_of_records=total_number_of_records+1

print("total_number_of_records="+str(total_number_of_records))
          #exit(0)
for roomId in mesh2irInputData :
              if  os.path.exists( mesh2ir_embeddings_dir+"/"+roomId+".pickle") :
                   os.remove(mesh2ir_embeddings_dir+"/"+roomId+".pickle")
              with open(mesh2ir_embeddings_dir+"/"+roomId+".pickle", 'wb') as f:
                   pickle.dump(mesh2irInputData[roomId], f, protocol=2)
                   f.close()


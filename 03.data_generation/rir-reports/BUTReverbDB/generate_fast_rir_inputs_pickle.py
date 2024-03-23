#!/usr/bin/env python3
from RIRHeader import *
from RIRLogger import *
from RIRData import *
from RIRDiffMethodEvaluator import *
from RIRReportGenerator import *

from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)


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




def permuteToFastRIRGANInput(x,y,z,roomDepth,roomWidth,roomHeight):
#     return y,roomWidth-z,roomHeight-x
     return x,roomWidth-y,z
  



def generateFastRIRInputs(data_dir):

          rir_data_file_path=data_dir+"/RIR.pickle.dat"
          fast_rir_data_input_file_path=data_dir+"/RIR.fast_rir_inputs.pickle"
   
          rir_data=[]  ##  "RIR.dat" --> list of list [34]
 
          with open(rir_data_file_path, 'rb') as f:
               rir_data=pickle.load(f)
          
          if  os.path.exists( fast_rir_data_input_file_path) :
              os.remove(fast_rir_data_input_file_path) 

          max_dimension = 5

          fastRirInputData=[]

          #counter=0, ## data limiter counter to test methods
          for dataline in rir_data:
              #counter+=1
              #if counter>5:
              #   break
              CENT=100 ## M / CM 
          
              roomDepth=float(dataline[int(rir_data_field_numbers['roomDepth'])])/CENT # CM to M
              roomWidth=float(dataline[int(rir_data_field_numbers['roomWidth'])])/CENT # CM to M
              roomHeight=float(dataline[int(rir_data_field_numbers['roomHeight'])])/CENT # CM to M
                  
              microphoneCoordinatesX=float(dataline[int(rir_data_field_numbers['microphoneStandInitialCoordinateX'])])/CENT +float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateX'])])/CENT # CM to M 
              microphoneCoordinatesY=float(dataline[int(rir_data_field_numbers['microphoneStandInitialCoordinateY'])])/CENT +float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateY'])])/CENT # CM to M
              microphoneCoordinatesZ=float(dataline[int(rir_data_field_numbers['mic_RelativeCoordinateZ'])])/CENT

              microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ=permuteToFastRIRGANInput(microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ,roomDepth,roomWidth,roomHeight)

              speakerCoordinatesX=float(dataline[int(rir_data_field_numbers['speakerStandInitialCoordinateX'])])/CENT +float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateX'])])/CENT # CM to M
              speakerCoordinatesY=float(dataline[int(rir_data_field_numbers['speakerStandInitialCoordinateY'])])/CENT +float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateY'])])/CENT # CM to M
              speakerCoordinatesZ=float(dataline[int(rir_data_field_numbers['speakerRelativeCoordinateZ'])])/CENT

              speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ=permuteToFastRIRGANInput(speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ,roomDepth,roomWidth,roomHeight)
              rt60=float(dataline[int(rir_data_field_numbers['rt60'])])




              speakerMotorIterationNo=int(dataline[int(rir_data_field_numbers['speakerMotorIterationNo'])])
              microphoneMotorIterationNo=int(dataline[int(rir_data_field_numbers['microphoneMotorIterationNo'])])
              currentActiveSpeakerNo=int(dataline[int(rir_data_field_numbers['currentActiveSpeakerNo'])])
              currentActiveSpeakerChannelNo=int(dataline[int(rir_data_field_numbers['currentActiveSpeakerChannelNo'])])
              physicalSpeakerNo=int(dataline[int(rir_data_field_numbers['physicalSpeakerNo'])]) 
              roomId=dataline[int(rir_data_field_numbers['roomId'])] 
              configId=dataline[int(rir_data_field_numbers['configId'])] 
              micNo=dataline[int(rir_data_field_numbers['micNo'])] 



              # https://github.com/anton-jeran/FAST-RIR
              # Listener Position = LP
              # Source Position = SP
              # Room Dimension = RD
              # Reverberation Time = T60
              # Correction = CRR

              # CRR = 0.1 if 0.5<T60<0.6
              # CRR = 0.2 if T60>0.6
              # CRR = 0 otherwise

              #Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_Y,SP_Z,RD_X,RD_Y,RD_Z,(T60+CRR)] /5) - 1

              CRR=0
              if 0.5 < rt60 < 0.6 :
                 CRR=0.1
              elif 0.6 < rt60:
                 CRR=0.2
                    
              #fastRirDataline=[microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ,speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ,roomDepth,roomWidth,roomHeight,rt60+CRR]
              fastRirDataline=[microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ,speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ,roomDepth,roomWidth,roomHeight,rt60+CRR]
               
               
              #print(fastRirDataline)
               
              fastRirDataline =np.divide(fastRirDataline,max_dimension)-1
              
              fastRirDataline=np.append(fastRirDataline,roomId)
              fastRirDataline=np.append(fastRirDataline,configId)
              fastRirDataline=np.append(fastRirDataline,speakerMotorIterationNo)
              fastRirDataline=np.append(fastRirDataline,microphoneMotorIterationNo)
              fastRirDataline=np.append(fastRirDataline,physicalSpeakerNo)
              fastRirDataline=np.append(fastRirDataline,micNo)
              #fastRirDataline=fastRirDataline/2
             
              
              #fastRirDatalineAsStr=s = "_".join([str(i) for i in fastRirDataline])
              essFilePath=dataline[int(rir_data_field_numbers['essFilePath'])]

              if  True :
             
#              if  6 <= roomDepth and roomDepth <= 12 and \
#                  8 <= roomWidth and roomWidth <= 12 and \
#                  2.5 <= roomHeight and roomHeight <= 3.5 and \
#                  1 <= microphoneCoordinatesX and microphoneCoordinatesX <= roomDepth  and \
#                  1 <= microphoneCoordinatesY and microphoneCoordinatesY <= roomWidth  and \
#                  1 <= speakerCoordinatesX and speakerCoordinatesX <= roomDepth and \
#                  1 <= speakerCoordinatesY and speakerCoordinatesY <= roomWidth and \
#                  0.2 <= rt60 and rt60 <= 2:
                  
#              if  6 <= roomDepth and roomDepth <= 10.5 and \
#                  6 <= roomWidth and roomWidth <= 10.5 and \
#                  2.5 <= roomHeight and roomHeight <= 3.5 and \
#                  6 <= microphoneCoordinatesX and microphoneCoordinatesX <= 10.5 and \
#                  2 <= microphoneCoordinatesY and microphoneCoordinatesY <= 8 and \
#                  8 <= speakerCoordinatesX and speakerCoordinatesX <= 10 and \
#                  6 <= speakerCoordinatesY and speakerCoordinatesY <= 9 and \
#                  0.5 <= rt60 and rt60 <= 1.0 :

                 fastRirInputData.append(fastRirDataline)

              


                 #if 2 > rt60 and rt60 > 1.5 :
                 #   print(fastRirDataline)
                 #   print(essFilePath)


          
          #for i in range(len(fastRirInputData)):
          #   print(f"fastRirInputData[{i}]={(fastRirInputData[i]+1)*5}")
              
          print("len(fastRirDataline)="+str(len(fastRirInputData)))
          #exit(0)
          
          with open(fast_rir_data_input_file_path, 'wb') as f:
              pickle.dump(fastRirInputData, f, protocol=2)     
                   




def buildReverbDBRIRData(data_dir):

     rir_data=[]
     if not os.path.exists(data_dir+'/reverbdb.csv') :
        print (data_dir+"/reverbdb.csv file does not exist ...")
        exit(1)
     reverbdb_csv_data=np.array(np.loadtxt(open(data_dir+'/reverbdb.csv', 'r'), delimiter=",",dtype=str))

     print ("Building  RIR.pickle.dat ...")
     for csvRowNumber in range(1,int(reverbdb_csv_data.shape[0])) :
            if csvRowNumber%100 == 0 :
               print(f"csvRowNumber={csvRowNumber}")
            essDataLine=reverbdb_csv_data[csvRowNumber].tolist()
            #print("len(essDataLine)="+str(len(essDataLine)))
            essFilePath=essDataLine[int(rir_data_field_numbers["essFilePath"])]
            wavRIRData,sampling_rate=librosa.load(data_dir+"/"+essFilePath,sr=16000)
            t60_impulse=compute_t60_impulse(wavRIRData,sampling_rate)
            essDataLine.append(t60_impulse)
            essDataLine.append(wavRIRData)
            if t60_impulse < 2 :
               rir_data.append(essDataLine)

     if len(rir_data) > 0 :
            rir_data_file=open(data_dir+"/RIR.pickle.dat",'wb')
            pickle.dump(rir_data,rir_data_file)
            rir_data_file.close()

     print(f"len(rir_data[0]) = {len(rir_data[0])}")
     print(f"len(rir_data) = {len(rir_data)}")

def compute_t60_impulse(raw_signal,fs):  # pylint: disable=too-many-locals
    bands =np.array([62.5 ,125, 250, 500,1000, 2000])

    if np.max(raw_signal)==0 and np.min(raw_signal)==0:
        print('came 1')
        return .5
    band_type = _check_band_type(bands)
    low = octave_low(bands[0], bands[-1])
    high = octave_high(bands[0], bands[-1])
    init = -0.0
    end = -60.0
    factor = 1.0
    bands =bands[3:5]
    low = low[3:5]
    high = high[3:5]

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))
        if math.isnan(sch_db[1]):
            print('came 2')
            return .5
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    mean_t60 =(t60[1]+t60[0])/2
    if math.isnan(mean_t60):
        print('came 3')
        return .5
    return mean_t60



def main(rir_data_path):
  print("REVERDB.CSV to DATA PICKLE FILE GENERATION IS STARTED.")
  buildReverbDBRIRData(rir_data_path)
  print("REVERDB.CSV to DATA PICKLE FILE GENERATION IS FINISHED.")
  print("FAST RIR INPUT PICKLE FILE GENERATION IS STARTED.")
  generateFastRIRInputs(rir_data_path)
  print("FAST RIR INPUT PICKLE FILE GENERATION IS FINISHED.")
  
                   

if __name__ == '__main__':
 main(str(sys.argv[1]).strip())

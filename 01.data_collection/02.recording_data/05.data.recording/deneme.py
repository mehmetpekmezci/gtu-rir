#!/usr/bin/env python3
from header import *
from microphone_numbers import * 
from speaker_numbers import * 
from Record import *
from ess import *
from transmitter import *
from receiver import *
from sensor import *
from webcam import *
from motor import *
from save import *

R_SPEAKER.append(20)
R_SPEAKER.append(30)
R_SPEAKER.append(40)
R_SPEAKER.append(50)

Z_SPEAKER.append(60)
Z_SPEAKER.append(70)
Z_SPEAKER.append(80)
Z_SPEAKER.append(90)

R_MIC.append(200)
R_MIC.append(300)
R_MIC.append(400)
R_MIC.append(500)
R_MIC.append(600)
R_MIC.append(700)

Z_MIC.append(800)
Z_MIC.append(900)
Z_MIC.append(1000)
Z_MIC.append(1100)
Z_MIC.append(1200)
Z_MIC.append(1300)

def main():
              essSignal=generate_left_ess_signal()
              record=Record() 
              record.timestamp='1'
              record.roomNo='2'
              record.configNo='2'
              record.speakerMotorIterationNo='3'
              record.microphoneMotorIterationNo='4'
              record.speakerMotorIterationDirection='1'
              record.microphoneStandInitialCoordinate=[ 1 , 2, 3 ]
              record.speakerStandInitialCoordinate=[ 4, 5, 6 ]
              record.currentActiveSpeakerNo=0
              record.currentActiveSpeakerChannelNo=1
              record.physicalSpeakerNo=record.getSpeakerNo()
              #record.photos=takePhoto()
              #print('A')
              # THESE records will be taken from /tmp/tempHum.txt by save.py
              #tempHum=getMicrophoneTemperatureHumidity()
              #record.temperatureAtMicrohponeStand=tempHum[0]
              #record.humidityAtMicrohponeStand=tempHum[1]
              #tempHum=getSpeakerTemperatureHumidity()
              #record.temperature=tempHum[0]
              #record.humidity=tempHum[1]
              record.transmittedSignal=essSignal
             
              threads=[] 
             
              for microphoneNo in range(NUMBER_OF_MICROPHONES):
                record.appendReceivedSignal(microphoneNo,[1,1,1,1])
              #   t = threading.Thread(target=receiveESSSignal, args = (microphoneNo, record))
              #   threads.append(t)
         
              #for microphoneNo in range(NUMBER_OF_MICROPHONES):
              #   threads[microphoneNo].start()
           
              #transmitESSSignal(0,essSignal)
        
              #for microphoneNo in range(NUMBER_OF_MICROPHONES):
              #   threads[microphoneNo].join()
                                
              save(record)


if __name__ == '__main__':
 main()





  

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


def main():

   ess_signal=generate_ess_signal()
   
   leftEssSignal=generate_left_signal(ess_signal)
   transmitSignal(0,leftEssSignal,"TEST_MODE_NONE")
   rightEssSignal=generate_right_signal(ess_signal)
   transmitSignal(0,rightEssSignal,"TEST_MODE_NONE")
   

   hour=datetime.datetime.now().hour
   while hour < START_HOUR:
      hour=datetime.datetime.now().hour
      logger.info(f"{hour} is not yet {START_HOUR}, script will start at approximately {START_HOUR}:00 - {START_HOUR}:10 ")
      time.sleep(600) ## sleep 600 seconds = 10 mins

   
       
   for speakerIterationNo in range(MAX_NUMBER_OF_SPEAKER_ITERATION):
       
       logger.info(">>Speaker Iteration No : "+str(speakerIterationNo))
       if speakerIterationNo%2 == 0 : 
          microphoneIterationDirection=1
       else :
          microphoneIterationDirection=0 
       
       for microphoneIterationNo in range(MAX_NUMBER_OF_MIC_ITERATION):
            logger.info(">>>Microphone Iteration No : "+str(microphoneIterationNo))

            for activeSpeakerNo in range(len(SPEAKERS)):
             logger.info(">>>>Active Speaker No : "+str(activeSpeakerNo))
             for channelNo in range(2):
              #logger.info(">>>> Reset USB Ports  ...")
              #process=subprocess.Popen([SCRIPT_DIR+"/reset_usb_ports_if_test_devices_fail.sh"],shell=True,stdout=subprocess.PIPE)
              #out,err=process.communicate()
              #print(out)
              #print(err)

              logger.info(">>>>Channel No : "+str(channelNo))

              if channelNo == 0 :
                essSignal=leftEssSignal
                songSignal=leftSongSignal
              else : 
                essSignal=rightEssSignal
                songSignal=rightSongSignal
              record=Record()
              record.timestamp=str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))
              record.roomNo=room_number
              record.configNo="config1"
              record.speakerMotorIterationNo=speakerIterationNo
              record.microphoneMotorIterationNo=microphoneIterationNo
              record.speakerMotorIterationDirection=microphoneIterationDirection
              record.microphoneStandInitialCoordinate=initialMicrophoneStandCoordinates ## [ X , Y , Z ]
              record.speakerStandInitialCoordinate=initialSpeakerStandCoordinates ## [ X , Y , Z ]
              record.currentActiveSpeakerNo=activeSpeakerNo    
              record.currentActiveSpeakerChannelNo=channelNo    
              record.physicalSpeakerNo=record.getSpeakerNo()
              record.photos=takePhoto()
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
                 t = threading.Thread(target=receiveESSSignal, args = (microphoneNo, record,TEST_MODE))
                 threads.append(t)
         
              for microphoneNo in range(NUMBER_OF_MICROPHONES):
                 threads[microphoneNo].start()
           
              transmitSignal(activeSpeakerNo,essSignal,TEST_MODE)
        
              for microphoneNo in range(NUMBER_OF_MICROPHONES):
                 threads[microphoneNo].join()


              time.sleep(10) # sleep 10 seconds
              logger.info(">>>>>> Transmitting and Receiving Song ")



              record.transmittedSongSignal=songSignal
             
              threads=[] 
             
              for microphoneNo in range(NUMBER_OF_MICROPHONES):
                 t = threading.Thread(target=receiveSongSignal, args = (microphoneNo, record,TEST_MODE))
                 threads.append(t)
         
              for microphoneNo in range(NUMBER_OF_MICROPHONES):
                 threads[microphoneNo].start()
           
              transmitSignal(activeSpeakerNo,songSignal,TEST_MODE,format =pyaudio.paFloat32)
        
              for microphoneNo in range(NUMBER_OF_MICROPHONES):
                 threads[microphoneNo].join()

                                
              save(record)
              
              time.sleep(10) # sleep 10 seconds
             
            # same level as "for activeSpeakerNo in range(len(SPEAKERS)):" above.
            logger.info(">>>> Microphone Iteration Direction is : "+str(microphoneIterationDirection))
            moveMicophoneStepMotor(microphoneIterationDirection,microphoneIterationNo,TEST_MODE)
            logger.info(">>>> Sleep 10 seconds , Wait the step motor noise to fade out...")
            time.sleep(10) # sleep 10 seconds, wait the step motor noise to fade out.
     
       # same level as "for microphoneIterationNo in range(maxNumberOfMicrophoneIteration)" above.

       moveSpeakerStepMotor(speakerIterationNo,TEST_MODE)
       logger.info(">>>> Sleep 10 seconds , Wait the step motor noise to fade out...")
       time.sleep(10) # sleep 10 seconds, wait the step motor noise to fade out.
        


if __name__ == '__main__':
 main()





  

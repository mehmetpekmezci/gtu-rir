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

START_HOUR=0

initialMicrophoneStandCoordinates=[]
initialSpeakerStandCoordinates=[]


def loadSpeakerAndMicrophonePositionsRelativeToTheirOwnStand():

   ## KEY'LER KUCUK HARF OLMALIDIR !!!!
   config = configparser.ConfigParser()
   

   if os.path.exists(DATA_DIR+"/single-speaker/setup-properties/record.ini") :
      setupState=input("Is there any change in Microphone and Speaker relative positions to their own stands ? (y/n)")
      if setupState == "y" :
        os.replace(DATA_DIR+"/single-speaker/setup-properties/record.ini",DATA_DIR+"/single-speaker/setup-properties/record.ini.org")
        print("copied old config to : "+ DATA_DIR+"/single-speaker/setup-properties/record.ini.org")
        os.remove(DATA_DIR+"/single-speaker/setup-properties/record.ini") 
      
   if not os.path.exists(DATA_DIR+"/single-speaker/setup-properties/record.ini"):

        if not os.path.exists(DATA_DIR+"/single-speaker/setup-properties"):
           os.makedirs(DATA_DIR+"/single-speaker/setup-properties")
       

        config['speaker.positions']={}
        for speakerNo in range(NUMBER_OF_SPEAKERS):
            while True:
              r = input(f"speaker_{speakerNo}_R (Centimeters) :")
              try:
                r = float(r)
                break
              except ValueError:
                print ('Numbers only')
            config['speaker.positions'][f'speaker_{speakerNo}_r']=str(r)
            while True:
              z = input(f"speaker_{speakerNo}_Z (Centimeters) :")
              try:
                z = float(z)
                break
              except ValueError:
                print ('Numbers only')
            config['speaker.positions'][f'speaker_{speakerNo}_z']=str(z)

        config['mic.positions']={}
        for micNo in range(NUMBER_OF_MICROPHONES):
            while True:
              r = input(f"mic_{micNo}_R (Centimeters) :")
              try:
                r = float(r)
                break
              except ValueError:
                print ('Numbers only')
            config['mic.positions'][f'mic_{micNo}_r']=str(r)
            while True:
              z = input(f"mic_{micNo}_Z (Centimeters) :")
              try:
                z = float(z)
                break
              except ValueError:
                print ('Numbers only')
            config['mic.positions'][f'mic_{micNo}_z']=str(z)

        with open(DATA_DIR+"/single-speaker/setup-properties/record.ini", 'w') as configfile:
             config.write(configfile)

   #### LOAD FROM FILE

   config.read(DATA_DIR+"/single-speaker/setup-properties/record.ini")

   
   for speakerNo in range(NUMBER_OF_SPEAKERS):
        R_SPEAKER.append(float(config['speaker.positions'][f'speaker_{speakerNo}_r']))
        Z_SPEAKER.append(float(config['speaker.positions'][f'speaker_{speakerNo}_z']))
        
   for micNo in range(NUMBER_OF_MICROPHONES):
        R_MIC.append(float(config['mic.positions'][f'mic_{micNo}_r']))
        Z_MIC.append(float(config['mic.positions'][f'mic_{micNo}_z']))




def loadRoomDimensions(room_number):

   global ROOM_DIM_WIDTH,ROOM_DIM_DEPTH,ROOM_DIM_HEIGHT

   ## KEY'LER KUCUK HARF OLMALIDIR !!!!
   config = configparser.ConfigParser()

      
   if not os.path.exists(DATA_DIR+"/single-speaker/room-"+str(room_number)+"/properties"):
        os.makedirs(DATA_DIR+"/single-speaker/room-"+str(room_number)+"/properties")

        config['room.dimensions']={}
        while True:
          room_width = input(f"room_width (Measured parallel to door face)  (Centimeters) :")
          try:
            room_width = float(room_width)
            break
          except ValueError:
            print ('Numbers only')
        config['room.dimensions'][f'room_width']=str(room_width)
       
        while True:
          room_depth = input(f"room_depth (Measured perpendicular to door face) (Centimeters) :")
          try:
            room_depth = float(room_depth)
            break
          except ValueError:
            print ('Numbers only')
        config['room.dimensions'][f'room_depth']=str(room_depth)
       
        while True:
          room_height = input(f"room_height (Height of the room) (Centimeters) :")
          try:
            room_height = float(room_height)
            break
          except ValueError:
            print ('Numbers only')
        config['room.dimensions'][f'room_height']=str(room_height)
       

        with open(DATA_DIR+"/single-speaker/room-"+str(room_number)+"/properties/record.ini", 'w') as configfile:
             config.write(configfile)

   #### LOAD FROM FILE

   config.read(DATA_DIR+"/single-speaker/room-"+str(room_number)+"/properties/record.ini")
   ROOM_DIM_WIDTH=float(config['room.dimensions']['room_width'])
   ROOM_DIM_DEPTH=float(config['room.dimensions']['room_depth'])
   ROOM_DIM_HEIGHT=float(config['room.dimensions']['room_height'])

def euclidean_distance(x1,y1,x2,y2):
    x_sqr=pow((x1-x2),2)
    y_sqr=pow((y1-y2),2)
    return math.sqrt(x_sqr+y_sqr)

def getStandPositions():
   logger.info("0,0                     y --> +                              ")
   logger.info(" #############################################################")
   logger.info(" #############################################################")
   logger.info(" #############################################################")
   logger.info("X  ###########################################################")
   logger.info("|  ###########################################################")
   logger.info("+  ###########################################################")
   logger.info(" #############################################################")
   logger.info(" #############################################################")
   logger.info(" #############################################################")
   logger.info(" #############################################################")
   logger.info(" #############################################################")
   logger.info(" ####################   DOOR       ###########################")

   initialMicrophoneStandCoordinatesX=-100000
   initialMicrophoneStandCoordinatesXInput=""
   R_OF_THE_MICROPHONE_CARRIER=150
   while initialMicrophoneStandCoordinatesX+ R_OF_THE_MICROPHONE_CARRIER > ROOM_DIM_DEPTH or  initialMicrophoneStandCoordinatesX - R_OF_THE_MICROPHONE_CARRIER < 0 :
         while True:
           initialMicrophoneStandCoordinatesXInput=input("Initial Microphone Stand Cooridinates X Centimeters :")
           try :
             initialMicrophoneStandCoordinatesX=float(initialMicrophoneStandCoordinatesXInput)
             if initialMicrophoneStandCoordinatesX + R_OF_THE_MICROPHONE_CARRIER > ROOM_DIM_DEPTH :
              logger.info (f'initialMicrophoneStandCoordinatesX + R_OF_THE_MICROPHONE_CARRIER ({initialMicrophoneStandCoordinatesX + R_OF_THE_MICROPHONE_CARRIER}) > ROOM_DIM_DEPTH ({ROOM_DIM_DEPTH}) , Please RE-MEASURE the microphone stand\'s X position')
             elif initialMicrophoneStandCoordinatesX - R_OF_THE_MICROPHONE_CARRIER < 0 :
              logger.info (f'initialMicrophoneStandCoordinatesX - R_OF_THE_MICROPHONE_CARRIER ({initialMicrophoneStandCoordinatesX - R_OF_THE_MICROPHONE_CARRIER}) < 0 , Please RE-MEASURE the microphone stand\'s X position')
             else:
               break
           except ValueError:
             print ('Numbers only')
   logger.info ('initialMicrophoneStandCoordinatesX is : '+str(initialMicrophoneStandCoordinatesX))
   initialMicrophoneStandCoordinates.append(initialMicrophoneStandCoordinatesX)
   
   initialMicrophoneStandCoordinatesY=-100000
   initialMicrophoneStandCoordinatesYInput=""
   while initialMicrophoneStandCoordinatesY+ R_OF_THE_MICROPHONE_CARRIER > ROOM_DIM_WIDTH or  initialMicrophoneStandCoordinatesY - R_OF_THE_MICROPHONE_CARRIER < 0 :
         while True:
           initialMicrophoneStandCoordinatesYInput=input("Initial Microphone Stand Cooridinates Y Centimeters :")
           try :
             initialMicrophoneStandCoordinatesY=float(initialMicrophoneStandCoordinatesYInput)
             if initialMicrophoneStandCoordinatesY + R_OF_THE_MICROPHONE_CARRIER > ROOM_DIM_WIDTH :
              logger.info (f'initialMicrophoneStandCoordinatesY + R_OF_THE_MICROPHONE_CARRIER ({initialMicrophoneStandCoordinatesY + R_OF_THE_MICROPHONE_CARRIER}) > ROOM_DIM_WIDTH ({ROOM_DIM_WIDTH}) , Please RE-MEASURE the microphone stand\'s Y position')
             elif initialMicrophoneStandCoordinatesY - R_OF_THE_MICROPHONE_CARRIER < 0 :
              logger.info (f'initialMicrophoneStandCoordinatesY - R_OF_THE_MICROPHONE_CARRIER ({initialMicrophoneStandCoordinatesY - R_OF_THE_MICROPHONE_CARRIER}) < 0 , Please RE-MEASURE the microphone stand\'s Y position')
             else:
              break
           except ValueError:
             print ('Numbers only')
   logger.info ('initialMicrophoneStandCoordinatesY is : '+str(initialMicrophoneStandCoordinatesY))
   initialMicrophoneStandCoordinates.append(initialMicrophoneStandCoordinatesY)

   initialMicrophoneStandCoordinatesZ=0
   initialMicrophoneStandCoordinates.append(initialMicrophoneStandCoordinatesZ)

   logger.info ('initialMicrophoneStandCoordinates[0] is : '+str(initialMicrophoneStandCoordinates[0]))
   logger.info ('initialMicrophoneStandCoordinates[1] is : '+str(initialMicrophoneStandCoordinates[1]))
   logger.info ('initialMicrophoneStandCoordinates[2] is : '+str(initialMicrophoneStandCoordinates[2]))
   
   
   
   
   

   initialSpeakerStandCoordinatesX=-100000
   initialSpeakerStandCoordinatesXInput=""
   R_OF_THE_SPEAKER_CARRIER=100
   R_BETWEEN_MICROPHONE_AND_SPAKER_STANDS=-10000
   while R_BETWEEN_MICROPHONE_AND_SPAKER_STANDS < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER :
    while initialSpeakerStandCoordinatesX+ R_OF_THE_SPEAKER_CARRIER > ROOM_DIM_DEPTH or  initialSpeakerStandCoordinatesX - R_OF_THE_SPEAKER_CARRIER < 0  :
         while True:
           initialSpeakerStandCoordinatesXInput=input("Initial Speaker Stand Cooridinates X Centimeters :")
           try :
             initialSpeakerStandCoordinatesX=float(initialSpeakerStandCoordinatesXInput)
             if initialSpeakerStandCoordinatesX + R_OF_THE_SPEAKER_CARRIER > ROOM_DIM_DEPTH :
              logger.info (f'initialSpeakerStandCoordinatesX + R_OF_THE_SPEAKER_CARRIER ({initialSpeakerStandCoordinatesX + R_OF_THE_SPEAKER_CARRIER}) > ROOM_DIM_DEPTH ({ROOM_DIM_DEPTH}) , Please RE-MEASURE the speaker stand\'s X position')
             elif initialSpeakerStandCoordinatesX - R_OF_THE_SPEAKER_CARRIER < 0 :
              logger.info (f'initialSpeakerStandCoordinatesX - R_OF_THE_SPEAKER_CARRIER ({initialSpeakerStandCoordinatesX - R_OF_THE_SPEAKER_CARRIER}) < 0 , Please RE-MEASURE the speaker stand\'s X position')
             #elif abs(initialSpeakerStandCoordinatesX-initialMicrophoneStandCoordinatesX) < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER  :
             # logger.info (f'abs(initialSpeakerStandCoordinatesX-initialMicrophoneStandCoordinatesX) ({abs(initialSpeakerStandCoordinatesX-initialMicrophoneStandCoordinatesX)}) < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER ({R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER}) , Please RE-MEASURE the speaker stand\'s X position')
             else :
              break
           except ValueError:
             print ('Numbers only')
   
    initialSpeakerStandCoordinatesY=-100000
    initialSpeakerStandCoordinatesYInput=""

    while initialSpeakerStandCoordinatesY+ R_OF_THE_SPEAKER_CARRIER > ROOM_DIM_WIDTH or  initialSpeakerStandCoordinatesY - R_OF_THE_SPEAKER_CARRIER < 0 :
         while True:
           initialSpeakerStandCoordinatesYInput=input("Initial Speaker Stand Cooridinates Y Centimeters :")
           try :
             initialSpeakerStandCoordinatesY=float(initialSpeakerStandCoordinatesYInput)
             if initialSpeakerStandCoordinatesY + R_OF_THE_SPEAKER_CARRIER > ROOM_DIM_WIDTH :
              logger.info (f'initialSpeakerStandCoordinatesY + R_OF_THE_SPEAKER_CARRIER ({initialSpeakerStandCoordinatesY + R_OF_THE_SPEAKER_CARRIER}) > ROOM_DIM_WIDTH ({ROOM_DIM_WIDTH}) , Please RE-MEASURE the speaker stand\'s Y position')
             if initialSpeakerStandCoordinatesY - R_OF_THE_SPEAKER_CARRIER < 0 :
              logger.info (f'initialSpeakerStandCoordinatesY - R_OF_THE_SPEAKER_CARRIER ({initialSpeakerStandCoordinatesY - R_OF_THE_SPEAKER_CARRIER}) < 0 , Please RE-MEASURE the speaker stand\'s Y position')
             elif euclidean_distance(initialSpeakerStandCoordinatesX,initialSpeakerStandCoordinatesY,initialMicrophoneStandCoordinatesX,initialMicrophoneStandCoordinatesY) < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER  :
              logger.info (f'euclidean_distance(initialSpeakerStandCoordinatesX,initialSpeakerStandCoordinatesY,initialMicrophoneStandCoordinatesX,initialMicrophoneStandCoordinatesY) < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER  ({R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER}) , Please RE-MEASURE the speaker stand\'s Y position')
             else:
               break
           except ValueError:
             print ('Numbers only')
    
    
    DX=abs(initialSpeakerStandCoordinatesX-initialMicrophoneStandCoordinatesX)
    DY=abs(initialSpeakerStandCoordinatesY-initialMicrophoneStandCoordinatesY)
    R_BETWEEN_MICROPHONE_AND_SPAKER_STANDS=math.sqrt(DX**2+DY**2)
    if R_BETWEEN_MICROPHONE_AND_SPAKER_STANDS < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER :
       initialSpeakerStandCoordinatesX=-100000
       initialSpeakerStandCoordinatesY=-100000
       logger.info (f'R_BETWEEN_MICROPHONE_AND_SPAKER_STANDS ({R_BETWEEN_MICROPHONE_AND_SPAKER_STANDS}) < R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER ({R_OF_THE_SPEAKER_CARRIER+R_OF_THE_MICROPHONE_CARRIER}) , Please RE-MEASURE the speaker stand\'s positions')
   
   
   logger.info ('initialSpeakerStandCoordinatesX is : '+str(initialSpeakerStandCoordinatesX))
   initialSpeakerStandCoordinates.append(initialSpeakerStandCoordinatesX)
   
   logger.info ('initialSpeakerStandCoordinatesY is : '+str(initialSpeakerStandCoordinatesY))
   initialSpeakerStandCoordinates.append(initialSpeakerStandCoordinatesY)
 
   initialSpeakerStandCoordinatesZ=0
   initialSpeakerStandCoordinates.append(initialSpeakerStandCoordinatesZ)

   logger.info ('initialSpeakerStandCoordinates[0] is : '+str(initialSpeakerStandCoordinates[0]))
   logger.info ('initialSpeakerStandCoordinates[1] is : '+str(initialSpeakerStandCoordinates[1]))
   logger.info ('initialSpeakerStandCoordinates[2] is : '+str(initialSpeakerStandCoordinates[2]))
   


def main():

   if len(sys.argv) > 0 :
      TEST_MODE=sys.argv[1]
   
   ess_signal=generate_ess_signal()
   #transmitSignal(0,ess_signal,"TEST_MODE_DEVICE_MOC")

   song_signal=get_song_signal()
   #transmitSignal(0,song_signal,"TEST_MODE_DEVICE_MOC",format =pyaudio.paFloat32)
   
   leftEssSignal=generate_left_signal(ess_signal)
   rightEssSignal=generate_right_signal(ess_signal)
   #transmitSignal(0,leftEssSignal,"TEST_MODE_DEVICE_MOC")
   
   
   leftSongSignal=generate_left_signal(song_signal)
   rightSongSignal=generate_right_signal(song_signal)
   #transmitSignal(0,leftSongSignal,"TEST_MODE_DEVICE_MOC",format =pyaudio.paFloat32)
   
   
      
   loadSpeakerAndMicrophonePositionsRelativeToTheirOwnStand()

   resetSpeakerStepMotor(TEST_MODE)
   resetMicrophoneStepMotor(TEST_MODE)
   
   



   room_number=input("Room Number:")
   while room_number=="":
      room_number=input("Room Number:")
   logger.info ('room number is : '+str(room_number))

   loadRoomDimensions(room_number)
   
   getStandPositions()

   logger.info ('MAX_NUMBER_OF_MIC_ITERATION is : '+str(MAX_NUMBER_OF_MIC_ITERATION))
   logger.info ('MAX_NUMBER_OF_SPEAKER_ITERATION is : '+str(MAX_NUMBER_OF_SPEAKER_ITERATION))
 
   config_number="micx-"+str(initialMicrophoneStandCoordinates[0])+"-micy-"+str(initialMicrophoneStandCoordinates[1])+"-spkx-"+str(initialSpeakerStandCoordinates[0])+"-spky-"+str(initialSpeakerStandCoordinates[1])+"-"+str(RECORD_TIMESTAMP) 
   
   logger.info("##############################################################")
   logger.info(" Room        : "+str(room_number))
   logger.info(" Config      : "+str(config_number))
   
              
   logger.info("##############################################################")

   hour=datetime.datetime.now().hour
   while hour < START_HOUR:
      hour=datetime.datetime.now().hour
      logger.info(f"{hour} is not yet {START_HOUR}, script will start at approximately {START_HOUR}:00 - {START_HOUR}:10 ")
      time.sleep(600) ## sleep 600 seconds = 10 mins

   
       
   for microphoneIterationNo in range(MAX_NUMBER_OF_MIC_ITERATION):
       
       logger.info(">>Microphone Iteration No : "+str(microphoneIterationNo))
       if microphoneIterationNo%2 == 0 : 
          speakerIterationDirection=1
       else :
          speakerIterationDirection=0
       
       for speakerIterationNo in range(MAX_NUMBER_OF_SPEAKER_ITERATION):
            logger.info(">>>Speaker Iteration No : "+str(speakerIterationNo))

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
              record.configNo=config_number
              record.speakerMotorIterationNo=speakerIterationNo
              record.microphoneMotorIterationNo=microphoneIterationNo
              record.speakerMotorIterationDirection=speakerIterationDirection
              record.microphoneStandInitialCoordinate=initialMicrophoneStandCoordinates ## [ X , Y , Z ]
              record.speakerStandInitialCoordinate=initialSpeakerStandCoordinates ## [ X , Y , Z ]
              record.currentActiveSpeakerNo=activeSpeakerNo    
              record.currentActiveSpeakerChannelNo=channelNo    
              record.physicalSpeakerNo=record.getSpeakerNo()
              record.photos=takePhoto(TEST_MODE)
              record.microphoneMotorPosition=getMicrophoneMotorPosition(TEST_MODE)
              record.speakerMotorPosition=getSpeakerMotorPosition(TEST_MODE)

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


              if TEST_MODE == "TEST_MODE_NONE" :
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

              if TEST_MODE == "TEST_MODE_NONE" :
                 time.sleep(10) # sleep 10 seconds
             
            # same level as "for activeSpeakerNo in range(len(SPEAKERS)):" above.
            logger.info(">>>> Speaker Iteration Direction is : "+str(speakerIterationDirection))
            moveSpeakerStepMotor(speakerIterationDirection,speakerIterationNo,TEST_MODE)
            logger.info(">>>> Sleep 10 seconds , Wait the step motor noise to fade out...")
            if TEST_MODE == "TEST_MODE_NONE" :
               time.sleep(10) # sleep 10 seconds, wait the step motor noise to fade out.
     
       # same level as "for microphoneIterationNo in range(maxNumberOfMicrophoneIteration)" above.

       moveMicophoneStepMotor(microphoneIterationNo,TEST_MODE)
       #resetSpeakerStepMotor(TEST_MODE)
       logger.info(">>>> Sleep 10 seconds , Wait the step motor noise to fade out...")
       if TEST_MODE == "TEST_MODE_NONE" :
          time.sleep(10) # sleep 10 seconds, wait the step motor noise to fade out.
       else :
          time.sleep(1) # sleep 10 seconds, wait the step motor noise to fade out.
        


if __name__ == '__main__':
 main()





  

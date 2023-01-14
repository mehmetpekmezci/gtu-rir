#!/usr/bin/env python3
from header import *
import scipy.io.wavfile as wavfile

def save(record):
    CURRENT_DATA_DIR=DATA_DIR+"/single-speaker/room-"+str(record.roomNo)+"/"+str(record.configNo)+"/micstep-"+str(record.microphoneMotorIterationNo)+"-spkstep-"+str(record.speakerMotorIterationNo)+"-spkno-"+str(record.physicalSpeakerNo)
    logger.info ('------saving to : '+CURRENT_DATA_DIR )

    if not os.path.exists(CURRENT_DATA_DIR):
           os.makedirs(CURRENT_DATA_DIR)   

    tempHumTxt=open("/tmp/tempHum.txt","r")
    tempHumData=tempHumTxt.read()
    tempHumTxt.close()

    record_file = open(CURRENT_DATA_DIR+"/record.txt", "w")
    record_file.write("roomNo="+str(record.roomNo)+"\n")
    record_file.write("configNo="+str(record.configNo)+"\n")
    record_file.write("timestamp="+str(record.timestamp)+"\n")
    record_file.write("speakerMotorIterationNo="+str(record.speakerMotorIterationNo)+"\n")
    record_file.write("microphoneMotorIterationNo="+str(record.microphoneMotorIterationNo)+"\n")
    record_file.write("speakerMotorIterationDirection="+str(record.speakerMotorIterationDirection)+"\n")
    record_file.write("currentActiveSpeakerNo="+str(record.currentActiveSpeakerNo)+"\n")
    record_file.write("currentActiveSpeakerChannelNo="+str(record.currentActiveSpeakerChannelNo)+"\n")
    record_file.write("physicalSpeakerNo="+str(record.physicalSpeakerNo)+"\n")
    record_file.write("microphoneStandInitialCoordinateX="+str(record.microphoneStandInitialCoordinate[0])+"\n")
    record_file.write("microphoneStandInitialCoordinateY="+str(record.microphoneStandInitialCoordinate[1])+"\n")
    record_file.write("microphoneStandInitialCoordinateZ="+str(record.microphoneStandInitialCoordinate[2])+"\n")
    record_file.write("speakerStandInitialCoordinateX="+str(record.speakerStandInitialCoordinate[0])+"\n")
    record_file.write("speakerStandInitialCoordinateY="+str(record.speakerStandInitialCoordinate[1])+"\n")
    record_file.write("speakerStandInitialCoordinateZ="+str(record.speakerStandInitialCoordinate[2])+"\n")
    record_file.write("microphoneMotorPosition="+str(record.microphoneMotorPosition)+"\n")
    record_file.write("speakerMotorPosition="+str(record.speakerMotorPosition)+"\n")

      
    #record_file.write("temperatureAtMicrohponeStand="+str(record.temperatureAtMicrohponeStand)+"\n")
    #record_file.write("humidityAtMicrohponeStand="+str(record.humidityAtMicrohponeStand)+"\n")
    #record_file.write("temperatureAtMSpeakerStand="+str(record.temperatureAtMSpeakerStand)+"\n")
    #record_file.write("humidityAtSpeakerStand="+str(record.humidityAtSpeakerStand)+"\n")
    record_file.write(tempHumData+"\n")
    
    speakerRelativePosition=record.getRelativeSpeakerPosition()
    record_file.write("speakerRelativeCoordinateX="+str(speakerRelativePosition[0])+"\n")
    record_file.write("speakerRelativeCoordinateY="+str(speakerRelativePosition[1])+"\n")
    record_file.write("speakerRelativeCoordinateZ="+str(speakerRelativePosition[2])+"\n")
    
   
   
    micRelativePosition,mic_angles_Theta_Phi=record.getRelativeMicrophonePositionsAndAngles()
    for i in range(NUMBER_OF_MICROPHONES): 
      record_file.write(f"mic_{i}_RelativeCoordinateX="+str(micRelativePosition[i][0])+"\n")
      record_file.write(f"mic_{i}_RelativeCoordinateY="+str(micRelativePosition[i][1])+"\n")
      record_file.write(f"mic_{i}_RelativeCoordinateZ="+str(micRelativePosition[i][2])+"\n")
      record_file.write(f"mic_{i}_DirectionX="+str(MICROPHONE_DIRECTION_PROPERTIES[i]["x"])+"\n")
      record_file.write(f"mic_{i}_DirectionY="+str(MICROPHONE_DIRECTION_PROPERTIES[i]["y"])+"\n")
      record_file.write(f"mic_{i}_DirectionZ="+str(MICROPHONE_DIRECTION_PROPERTIES[i]["z"])+"\n")
      record_file.write(f"mic_{i}_Theta="+str(mic_angles_Theta_Phi[i][0])+"\n")
      record_file.write(f"mic_{i}_Phi="+str(mic_angles_Theta_Phi[i][1])+"\n")
      
    record_file.write("microphoneStandAngle="+str(record.microphoneStandAngle)+"\n")
    record_file.write("speakerStandAngle="+str(record.speakerStandAngle)+"\n")
    record_file.write("speakerAngleTheta="+str(record.speakerAngleTheta)+"\n")
    record_file.write("speakerAnglePhi="+str(record.speakerAnglePhi)+"\n")
    
    
    record_file.close()

    subprocess.Popen(["cat /proc/asound/cards >"+CURRENT_DATA_DIR+"/proc_asound_cards.txt"],shell=True)
    subprocess.Popen(["lsusb -t | grep -i 'usbhid\|hub' >"+CURRENT_DATA_DIR+"/lsusb-t.txt"],shell=True)

    for i in range(len(record.photos)):
      try:
           cv2.imwrite(CURRENT_DATA_DIR+'/microphone_and_speaker_stand_positions.'+str(i)+'.jpg',record.photos[i])   
      except:
           logger.error ('Error Saving Photo Image ...  ')
           

    logger.info ('------saving : len(record.transmittedSignal) = '+str(len(record.transmittedSignal))+'   len(record.receivedSignal)='+str(len(record.receivedSignal)))
    logger.info ('------saving : len(record.transmittedSignal) = '+str(len(record.transmittedSignal))+'   len(record.receivedSignal[1])='+str(len(record.receivedSignal[1])))


    if not os.path.exists(CURRENT_DATA_DIR+'/../../../transmittedEssSignal.wav'):
       wavfile.write(CURRENT_DATA_DIR+'/../../../transmittedEssSignal.wav',SAMPLE_RATE,record.transmittedSignal.astype(np.float32))
    
    if not os.path.exists(CURRENT_DATA_DIR+'/../../../transmittedSongSignal.wav'):
       wavfile.write(CURRENT_DATA_DIR+'/../../../transmittedSongSignal.wav',SAMPLE_RATE,record.transmittedSongSignal.astype(np.float32))
    
    for microphoneNo in range(NUMBER_OF_MICROPHONES):
       wavfile.write(CURRENT_DATA_DIR+'/receivedEssSignal-'+str(microphoneNo)+'.wav',SAMPLE_RATE,np.array(record.receivedSignal[microphoneNo]).astype(np.float32))
       
    for microphoneNo in range(NUMBER_OF_MICROPHONES):
       wavfile.write(CURRENT_DATA_DIR+'/receivedSongSignal-'+str(microphoneNo)+'.wav',SAMPLE_RATE,np.array(record.receivedSongSignal[microphoneNo]).astype(np.float32))


    subprocess.Popen(["bzip2 "+CURRENT_DATA_DIR+"/*.wav"],shell=True)


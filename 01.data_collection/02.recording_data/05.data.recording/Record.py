#!/usr/bin/env python3
from header import *

# this definition is also written in header.py
## ABSOLUTE COORDINATE DEFINITIONS ACCORDING TO THE ROOM
# [0,0,0] is the corner where the floor corner of the room closest to the door.
########################
##     Z
##     |__ Y
##     /       CENTER=Center of the microphone stand
##    X
#######################

########################
##     Z
##     | \
##     |  \              C= CENTER=Center of the microphone stand
##     |   \   M         M= Mikrophone coordinates
##     |   /|            M1= Projection of point M on XY plane
##     |P / |            P= PHI = ZCM angle 
##   C |_/__|______ Y    T= THETA = YCM1 angle
##     / \T |            
##    /   \ |      
##   /     \|  
##  /        M1
## X
#######################


class Record :

 def __init__(self):
      self.roomNo=-1
      self.configNo=-1
      self.speakerMotorIterationNo=-1
      self.microphoneMotorIterationNo=-1
      self.speakerMotorIterationDirection=-1
      self.currentActiveSpeakerNo=-1
      self.currentActiveSpeakerChannelNo=-1
      self.physicalSpeakerNo=-1
      self.timestamp=None       
      self.photos=[]
      self.transmittedSignal=[] ## [10x44100] = [ oss sginal length in seconds X samlpe rate ]
      self.receivedSignal=dict() ## microphoneNo x receivedEssSignal ,  [6] x [10x44100] data points.
      self.transmittedSongSignal=[] ## [10x44100] = [ oss sginal length in seconds X samlpe rate ]
      self.receivedSongSignal=dict() ## microphoneNo x receivedEssSignal ,  [6] x [10x44100] data points.
      self.temperature=None
      self.humidity=None
      self.microphoneStandInitialCoordinate=[] ## [ X , Y , Z ]
      self.speakerStandInitialCoordinate=[] ## [ X , Y , Z ] 
      self.microphoneMotorPosition=None
      self.speakerMotorPosition=None
      
      self.microphoneStandAngle=None ## Angle between the (MicrophoneStand-Microphone0) line  and the Y axis of the room
      self.speakerStandAngle=None ## Angle between (SpeakerStand-Speaker0) line and the Y axis of the room
      self.speakerAngleTheta=None ## Angle between (SpeakerStand-Speaker0) line and the Y axis of the room  in XY plane
      self.speakerAnglePhi=90 ## Angle between  (SpeakerStand-Speaker0) line  and the Z axis of the room in XZ plane
 
 def appendReceivedSignal(self,microphoneNo,audioArray):
     self.receivedSignal[ microphoneNo ] = audioArray
 
 def appendReceivedSongSignal(self,microphoneNo,audioArray):
     self.receivedSongSignal[ microphoneNo ] = audioArray
 
 def getSpeakerNo(self):
  speakerNo=-1
  if self.currentActiveSpeakerNo == 0 and self.currentActiveSpeakerChannelNo ==0  : speakerNo=0
  elif  self.currentActiveSpeakerNo == 0 and self.currentActiveSpeakerChannelNo ==1  : speakerNo=1
  elif  self.currentActiveSpeakerNo == 1 and self.currentActiveSpeakerChannelNo ==1  : speakerNo=2
  elif  self.currentActiveSpeakerNo == 1 and self.currentActiveSpeakerChannelNo == 0  : speakerNo=3
  #self.physicalSpeakerNo=speakerNo
  return speakerNo
    
 def getRelativeSpeakerPosition(self):
      speakerNo=int(self.getSpeakerNo())
      DIRECTION=float(self.speakerMotorIterationDirection)
      
      ### DIRECTION 1 is clockwise, 0 is counterclockwise  --> OPPOSITE of MICROPHONE STAND

      # [ X, Y, Z ]
      ## TAM TUR 200 birim
      ANGLE=(float(self.speakerMotorPosition)%200)/200*360
      
      self.speakerStandAngle=ANGLE ## Angle between (SpeakrStandCenter-Speaker0) line and the Y axis of the room
      self.speakerAngleTheta=ANGLE+90 ## ROLL, SPEKAERS are transmitting sound at this DIRECTION on XY Plane ( Y Axis is reference)
      self.speakerAnglePhi=90 ## PITCH, SPEKAERS are transmitting sound at this DIRECTION on XZ Plane ( Z Axis is reference)
                                              
      if speakerNo > 1 :  
          ANGLE=ANGLE+180 
      ANGLE=math.pi*ANGLE/180
      return [ R_SPEAKER[speakerNo]*math.sin(ANGLE) , R_SPEAKER[speakerNo]*math.cos(ANGLE) , Z_SPEAKER[speakerNo] ]

 def getRelativeMicrophonePositionsAndAngles(self):
      #MIC_STEP=float(self.microphoneMotorIterationNo)
      DIRECTION=1
      mic_physical_positions_according_to_stand=[]   
      mic_angles_Theta_Phi=[]   ## ROLL,PITCH
      # [ X, Y, Z ]

      #ANGLE=ONE_STEP_DEGREE*MIC_STEP
        
      ANGLE=(float(self.microphoneMotorPosition)%200)/200*360
      
      ## Michrophone direction starts with counterclockwise, which is opposie to the coordinate system of the room.(all rooms will have the same coordinate system), 
      # we ensure the starting point by facing the "green sticker of the microphone stand"   to    "the door of the room" .
      # --> Why microphone stand goes counterclockwise (opposite of the speaker stand) :  Because we soldered the motor wires oppositely to controller card (not intentionally :) )
      ANGLE=-ANGLE
      
      self.microphoneStandAngle=ANGLE ##  Angle between the (MicrophoneStandCenter-Microphone0) line  and the Y axis of the room


      for i in range(NUMBER_OF_MICROPHONES):
        THETA=ANGLE
        PHI=0


        if MICROPHONE_DIRECTION_PROPERTIES[i]['y'] == 1 :
           THETA=0+THETA
        elif    MICROPHONE_DIRECTION_PROPERTIES[i]['y'] == -1 :
            THETA=180+THETA
        else :
          if MICROPHONE_DIRECTION_PROPERTIES[i]['x'] == 1 :
           THETA=90+THETA
          elif    MICROPHONE_DIRECTION_PROPERTIES[i]['x'] == -1 :
            THETA=-90+THETA
          else :
            THETA=0+THETA

        if MICROPHONE_DIRECTION_PROPERTIES[i]['z'] == 1 :
           PHI=0
           THETA=0
        elif    MICROPHONE_DIRECTION_PROPERTIES[i]['z'] == -1 :
            PHI=180
            THETA=0
        else :
            PHI=90
            
        mic_angles_Theta_Phi.append([ THETA , PHI])

      
      #if DIRECTION == 1 :  ## DIRECTION 0 is clockwise, 1 is counterclockwise , starts with 1, ilk hareketini 4. bolgeye yapar.
      #   ANGLE=360-ANGLE
      
      ANGLE_PLUS_180=ANGLE+180 

      ANGLE=math.pi*ANGLE/180
      ANGLE_PLUS_180=math.pi*ANGLE_PLUS_180/180

      mic_physical_positions_according_to_stand.append([ R_MIC[0]*math.sin(ANGLE) , R_MIC[0]*math.cos(ANGLE) , Z_MIC[0]])
      mic_physical_positions_according_to_stand.append([ R_MIC[1]*math.sin(ANGLE) , R_MIC[1]*math.cos(ANGLE) , Z_MIC[1]])
      mic_physical_positions_according_to_stand.append([ R_MIC[2]*math.sin(ANGLE) , R_MIC[2]*math.cos(ANGLE) , Z_MIC[2]])
      mic_physical_positions_according_to_stand.append([ R_MIC[3]*math.sin(ANGLE_PLUS_180) , R_MIC[3]*math.cos(ANGLE_PLUS_180) , Z_MIC[3]])
      mic_physical_positions_according_to_stand.append([ R_MIC[4]*math.sin(ANGLE_PLUS_180) , R_MIC[4]*math.cos(ANGLE_PLUS_180) , Z_MIC[4]])
      mic_physical_positions_according_to_stand.append([ R_MIC[5]*math.sin(ANGLE_PLUS_180) , R_MIC[5]*math.cos(ANGLE_PLUS_180) , Z_MIC[5]])

      
      return mic_physical_positions_according_to_stand,mic_angles_Theta_Phi 


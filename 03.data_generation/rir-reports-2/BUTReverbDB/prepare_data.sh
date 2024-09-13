#!/bin/bash

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA
DEFAULT_MAIN_REPORT_DIR=$HOME/RIR_REPORT
REPORT_DIR=$DEFAULT_MAIN_REPORT_DIR/BUTReverbDB/

if [ "$BUTReverbDB_DATA" = "" ]
then
        export BUTReverbDB_DATA=$DEFAULT_MAIN_DATA_DIR/BUTReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only
        echo "BUTReverbDB_DATA=$BUTReverbDB_DATA"
fi


if [ ! -d $BUTReverbDB_DATA ]
then
     cd $DEFAULT_MAIN_DATA_DIR/BUTReverbDB
     wget http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz
     tar xvfz BUT_ReverbDB_rel_19_06_RIR-Only.tgz
else
     echo "Data directory $BUTReverbDB_DATA already exists , no need to recreate ."
fi

cd $BUTReverbDB_DATA

if [ ! -e reverbdb.csv ]
then

 timestamp=$(date '+%Y.%m.%d_%H.%M.%S')
 speakerMotorIterationNo=0
 microphoneMotorIterationNo=0
 speakerMotorIterationDirection=0
 currentActiveSpeakerNo=0
 currentActiveSpeakerChannelNo=0
 physicalSpeakerNo=0
 microphoneStandInitialCoordinateX=0
 microphoneStandInitialCoordinateY=0
 microphoneStandInitialCoordinateZ=0
 speakerStandInitialCoordinateX=0
 speakerStandInitialCoordinateY=0
 speakerStandInitialCoordinateZ=0
 microphoneMotorPosition=0
 speakerMotorPosition=0
 temperatureAtMicrohponeStand=0
 humidityAtMicrohponeStand=0
 temperatureAtMSpeakerStand=0
 humidityAtSpeakerStand=0
 tempHumTimestamp=$timestamp
 microphoneStandAngle=0
 speakerStandAngle=0
 speakerAngleTheta=0
 speakerAnglePhi=0
 mic_DirectionX=0
 mic_DirectionY=0
 mic_DirectionZ=0
 mic_Theta=0
 mic_Phi=0

 for roomId in *
 do
   if [ -d $roomId ]
   then
       configId=MicID01
       for spkDir in $roomId/$configId/*
       do
               speakerId=$(basename $spkDir)
               echo $speakerId
               if [ -d $spkDir ]
               then
                       for micDir in $roomId/$configId/$speakerId/*
                       do
                               if [ -d $micDir -a -e $micDir/RIR/*.v00.wav ]
                               then
                                       micNo=$(basename $micDir)
                                       filepath=$(ls $micDir/RIR/*.v00.wav| head -1)
                                       ls $filepath
				       
				       speakerRelativeCoordinateX=$(cat $micDir/mic_meta.txt| grep EnvSpk| grep -v Rel | grep Depth|awk '{print $2}'| head -1)
				       speakerRelativeCoordinateY=$(cat $micDir/mic_meta.txt| grep EnvSpk| grep -v Rel | grep Width|awk '{print $2}'| head -1)
				       speakerRelativeCoordinateZ=$(cat $micDir/mic_meta.txt| grep EnvSpk| grep -v Rel | grep Height|awk '{print $2}'| head -1)
				       mic_RelativeCoordinateX=$(cat $micDir/mic_meta.txt| grep EnvMic | grep -v Rel | grep Depth|awk '{print $2}'| head -1)
				       mic_RelativeCoordinateY=$(cat $micDir/mic_meta.txt| grep EnvMic | grep -v Rel | grep Width|awk '{print $2}'| head -1)
				       mic_RelativeCoordinateZ=$(cat $micDir/mic_meta.txt| grep EnvMic | grep -v Rel | grep Height|awk '{print $2}'| head -1)
				       roomDepth=$(cat $micDir/mic_meta.txt| grep EnvDepth|awk '{print $2}'| head -1)
				       roomWidth=$(cat $micDir/mic_meta.txt| grep EnvWidth|awk '{print $2}'| head -1)
				       roomHeight=$(cat $micDir/mic_meta.txt| grep EnvHeight|awk '{print $2}'| head -1)
				       physicalSpeakerNo=$speakerId
                                       echo "$timestamp,$speakerMotorIterationNo,$microphoneMotorIterationNo,$speakerMotorIterationDirection,$currentActiveSpeakerNo,$currentActiveSpeakerChannelNo,$physicalSpeakerNo,$microphoneStandInitialCoordinateX,$microphoneStandInitialCoordinateY,$microphoneStandInitialCoordinateZ,$speakerStandInitialCoordinateX,$speakerStandInitialCoordinateY,$speakerStandInitialCoordinateZ,$microphoneMotorPosition,$speakerMotorPosition,$temperatureAtMicrohponeStand,$humidityAtMicrohponeStand,$temperatureAtMSpeakerStand,$humidityAtSpeakerStand,$tempHumTimestamp,$speakerRelativeCoordinateX,$speakerRelativeCoordinateY,$speakerRelativeCoordinateZ,$microphoneStandAngle,$speakerStandAngle,$speakerAngleTheta,$speakerAnglePhi,$mic_RelativeCoordinateX,$mic_RelativeCoordinateY,$mic_RelativeCoordinateZ,$mic_DirectionX,$mic_DirectionY,$mic_DirectionZ,$mic_Theta,$mic_Phi,$filepath,$roomId,$configId,$micNo,$roomWidth,$roomHeight,$roomDepth" >> reverbdb.csv
                               fi
                       done
               fi
       done
   fi
 done
 perl -pi -e 's/,SpkID[0,1](.)_[0-9]{8}_[S,T],/,$1,/' reverbdb.csv
else
 echo "reverdb.csv file already exists ."
fi



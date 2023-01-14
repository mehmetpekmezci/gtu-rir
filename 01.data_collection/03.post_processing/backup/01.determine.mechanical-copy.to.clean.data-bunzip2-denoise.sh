#!/bin/bash




####
#### 1. DETERMINE THE MECHANICALLY PROBLEMATIC RECORDS BY LOOKING AT THE PHOTOS ( PHOTOS ARE ARRANGED IN SUCH A WAY THAT IT WILL BE EASY TO SEE IF THE RCORD HAS MECHANICAL PROBLEM)
#### 2. CREATE CLEAN_DATA_DIR AND COPY RELEVANT DATA INTO IT
#### 3. DECOMPRESS ALL THE BZ2 FILES IN CLAN DATA_DIR , DENOISE THE EXTRACTED WAV FILE
####




CURRENT_DIR=$(pwd)

DATA_DIR=$(realpath "../../../data/single-speaker/");
WORK_DIR="$HOME/work-rir"

SCRIPT_DIR=$(pwd)
CLEAN_DATA_DIR=$(realpath "../../../data/single-speaker-clean/");
if [ ! -e $CLEAN_DATA_DIR ]
then
	mkdir -p $CLEAN_DATA_DIR
fi

cd $DATA_DIR

#for i in $(find . -name 'iteration-?'); do   n=$(echo $i| cut -d- -f3) ; m=$(echo $i| cut -d- -f1,2);mv $i $m-0$n; done

if [ ! -f /var/tmp/photos_already_copied ]
then
mkdir -p $WORK_DIR

echo " START PHOTO INSPECTION (TO SEE IF ANY MECHANICAL PROBLEM OCCURED) "
echo "    COPYING PHOTOGRAPHS INTO A DIRECTORY (EASIER TO INSPECT) "
for roomconfig in $(ls | grep -v mic | grep -v work)
do  
       	echo "## Room Config :  $roomconfig  "; 
        for recorddate in $(ls $roomconfig)
	do
       	        echo "#### Record Date :  $recorddate  "

		mkdir -p $WORK_DIR/$roomconfig-$recorddate/

                for iteration in $(ls $roomconfig/$recorddate)
	        do
       	                echo "###### Iteration :  $iteration  "; 
                        for speakeriteration in $(ls $roomconfig/$recorddate/$iteration)
	                do
       	                      echo "######## Speaker Iteration :  $speakeriteration  "  >/dev/null
                              for microphoneiteration in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration)
	                      do
       	                             echo "########## Microphone Iteration :  $microphoneiteration  " >/dev/null
                                     for activespeakerno in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration)
	                             do
       	                                  echo "############ Active Speaker No :  $activespeakerno  " >/dev/null
                                          for activechannelno in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno)
	                                  do
       	                                      echo "############## Active Channel No :  $activechannelno  " > /dev/null 
                                              if [ ! -f $WORK_DIR/$roomconfig-$recorddate/$iteration-$speakeriteration-$microphoneiteration-$activespeakerno-$activechannelno.jpg -a -f $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno/*.jpg ] 
					      then
					           cp $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno/*.jpg $WORK_DIR/$roomconfig-$recorddate/$iteration-$speakeriteration-$microphoneiteration-$activespeakerno-$activechannelno.jpg
					      fi
					      #read
		                          done
		                     done
		              done
		        done
		done
	done
done

touch /var/tmp/photos_already_copied 

fi

if [ ! -f /var/tmp/filtered_out_mechanically_problematic_records ]
then


first_room_config=$(ls | grep -v mic | grep -v work| sort| head -1)
first_recorddate=$(ls $first_room_config/| grep -v mic | grep -v work| sort| head -1)


echo "******* INSPECT THE PHOTOS, DETERMINE IF THERE WERE ANY MECHANICAL CORRUPTION"
echo "******* ENTER INTO DIRECTORIES and DETERMINE IN WHICH STEP, MOTOR IS STOPPED" 
echo ""
echo ""
echo "******* ENTER INTO EACH DIRECTORY, DOUBLE-CLICK ON THE FIRST PHOTO,  GO ON UNTIL THE SPEAKER OR MICROPHONE STAND STOPS MOVING ..."
echo "******* CREATE BACKUP DIRECTORY, AND MOVE THE PROBLEMATIC PHOTOS INTO BACKUP DIRECTORY"
nautilus $WORK_DIR/ &


read

touch /var/tmp/filtered_out_mechanically_problematic_records 

fi



if [ ! -f /var/tmp/noise_already_cleaned ]
then

echo "COPYING  MECHANICALLY PROPER RECORDS INTO 'single-speaker-clean' DIRECTORY "

for i in $(ls $WORK_DIR/)
do
   room_config=$(echo $i | cut -d- -f1,2)
   record_date=$(echo $i | cut -d- -f3)
   for iteration in $(ls $WORK_DIR/$room_config-$record_date/| cut -d- -f1,2| sort| uniq )
   do
	   echo mkdir -p $CLEAN_DATA_DIR/$room_config/$record_date/
	   mkdir -p $CLEAN_DATA_DIR/$room_config/$record_date/
	   echo cp -Rf $DATA_DIR/$room_config/$record_date/$iteration $CLEAN_DATA_DIR/$room_config/$record_date/$iteration
	   cp -Rf $DATA_DIR/$room_config/$record_date/$iteration $CLEAN_DATA_DIR/$room_config/$record_date/$iteration
   done
done
touch /var/tmp/noise_already_cleaned
fi

echo "DECOMPRESSING WAV FILES AND REMOVING NOISE "

cd $CLEAN_DATA_DIR

for roomconfig in $(ls | grep -v mic | grep -v backup)
do  
       	echo "## Room Config :  $roomconfig  "; 
        for recorddate in $(ls $roomconfig)
	do
       	        echo "#### Record Date :  $recorddate  "

                for iteration in $(ls $roomconfig/$recorddate)
	        do
       	                echo "###### Iteration :  $iteration  "; 
                        for speakeriteration in $(ls $roomconfig/$recorddate/$iteration)
	                do
       	                      echo "######## Speaker Iteration :  $speakeriteration  " 
                              for microphoneiteration in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration)
	                      do
       	                             echo "########## Microphone Iteration :  $microphoneiteration  "
                                     for activespeakerno in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration)
	                             do
       	                                  echo "############ Active Speaker No :  $activespeakerno  "
                                          for activechannelno in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno)
	                                  do
       	                                      echo "############## Active Channel No :  $activechannelno  " 
                                              for recordfile in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno| grep wav.tar.bz2)
	                                      do
       	                                          echo "################ Record File :  $recordfile  " 
						  wavfile=$(echo $recordfile | sed -e 's/\.tar\.bz2//')
						  RECORD_DIR="$roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno"
                                                  if [ ! -f $RECORD_DIR/$wavfile ] 
					          then
						  (
							  echo tar xvjf $RECORD_DIR/$recordfile -C $RECORD_DIR
							  tar xvjf $RECORD_DIR/$recordfile -C $RECORD_DIR
							  FILESIZE=$(ls -l $RECORD_DIR/$wavfile  | awk '{print $5}')
							  if [ $FILESIZE -lt 1000000 ]
                                                          then
                                                                rm -f $RECORD_DIR/$wavfile
							  else
                                                                python3 $SCRIPT_DIR/noise_reducer.py $RECORD_DIR/$wavfile
								rm -f $RECORD_DIR/$wavfile
								mv $RECORD_DIR/$wavfile.clean.wav $RECORD_DIR/$wavfile
                                                        #       python3 $SCRIPT_DIR/alligner.py $RECORD_DIR/transmittedEssSignal.wav $RECORD_DIR/$wavfile 
							#	rm -f $RECORD_DIR/$wavfile
							#	mv $RECORD_DIR/$wavfile.alligned.wav $RECORD_DIR/$wavfile
							  fi	
							  rm -f $RECORD_DIR/$recordfile
					          ) &
					          fi
					      done
                                              for recordfile in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno| grep wav.bz2)
	                                      do
       	                                          echo "################ Record File :  $recordfile  " 
						  wavfile=$(echo $recordfile | sed -e 's/\.bz2//')
						  RECORD_DIR="$roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno"
                                                  if [ ! -f $RECORD_DIR/$wavfile ] 
					          then
					          (
							  echo bunzip2 $RECORD_DIR/$recordfile
							  bunzip2 $RECORD_DIR/$recordfile 
							  FILESIZE=$(ls -l $RECORD_DIR/$wavfile  | awk '{print $5}')
							  if [ $FILESIZE -lt 1000000 ]
                                                          then
                                                                rm -f $RECORD_DIR/$wavfile
							  else
                                                                python3 $SCRIPT_DIR/noise_reducer.py $RECORD_DIR/$wavfile
								rm -f $RECORD_DIR/$wavfile
								mv $RECORD_DIR/$wavfile.clean.wav $RECORD_DIR/$wavfile
                                                        #       python3 $SCRIPT_DIR/alligner.py $RECORD_DIR/transmittedEssSignal.wav $RECORD_DIR/$wavfile 
							#	rm -f $RECORD_DIR/$wavfile
							#	mv $RECORD_DIR/$wavfile.alligned.wav $RECORD_DIR/$wavfile
							  fi	
							  rm -f $RECORD_DIR/$recordfile
					          )&
					          fi
					      done
					      WAIT=1
					      while [ $WAIT == 1 ]
					      do
						      U=$(ps -ef | grep bz2| grep -v grep | wc -l )
						      P=$(ps -ef | grep python3| grep -v grep | grep noise_reducer | wc -l)
						      if [ $P -eq 0 -a $U -eq 0 ]
						      then
							      WAIT=0
						      else
							      echo "WAITING PARALLEL EXECUTION"
							      sleep 2
						      fi
				              done
		                          done
		                     done
		              done
		        done
		done
	done
done



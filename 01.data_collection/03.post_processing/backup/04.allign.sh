#!/bin/bash




####
#### ALLIGN RECEVIED WAV FILES WITH TRANSMITTED WAV FILE USING CROSS CORRELATION
####




CURRENT_DIR=$(pwd)

DATA_DIR=$(realpath "../../../data/single-speaker/");
WORK_DIR="$HOME/work-rir"

SCRIPT_DIR=$(pwd)
CLEAN_DATA_DIR=$(realpath "../../../data/single-speaker-clean/");

if [ ! -e $CLEAN_DATA_DIR/transmittedEssSignal.wav ]
then
    t=$(find $CLEAN_DATA_DIR -name transmittedEssSignal.wav | head -1)
    ln -s $t $CLEAN_DATA_DIR
fi

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
					      
                                              for wavfile in $(ls $roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno| grep receive| grep .wav | grep -v alligned)
	                                      do
						  RECORD_DIR="$roomconfig/$recorddate/$iteration/$speakeriteration/$microphoneiteration/$activespeakerno/$activechannelno"
                                                  if [ ! -f $RECORD_DIR/$wavfile.alligned ] 
					          then
						  (
       	                                                echo "################ Alligning Wav File :  $wavfile  " 
                                                       python3 $SCRIPT_DIR/alligner.py $RECORD_DIR/$wavfile $CLEAN_DATA_DIR/transmittedEssSignal.wav
							rm -f $RECORD_DIR/$wavfile
							mv $RECORD_DIR/$wavfile.alligned.wav $RECORD_DIR/$wavfile
							touch $RECORD_DIR/$wavfile.alligned
					          ) &
					          fi
					      done
					      WAIT=1
					      while [ $WAIT == 1 ]
					      do
						      U=$(ps -ef | grep bz2| grep -v grep | wc -l )
						      P=$(ps -ef | grep python3| grep -v grep | grep alligner.py | wc -l)
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



#!/bin/bash

SCRIPT_DIR=$(realpath $(pwd))


DATA_DIR=$(realpath "../../../data/single-speaker/");
cd $DATA_DIR
for i in $(find room* -name rec*Signal-*.wav.bz2 -size -1000k | grep -v trash | grep -v '\.ir\.wav'); do   echo mv $i $i.corrupted;mv $i $i.corrupted;  done

function delete_record {
    room=$1
    config=$2
    record=$3
    echo "delete_record $1 $2 $3"
    mkdir -p $room/$config/trash
    record_base=$(echo $record | sed -e 's/spkno-0/spkno/')
    echo "mv $room/$config/$record_base-* $room/$config/trash" 
    mv $room/$config/$record_base-* $room/$config/trash
}

function snr_based_delete_and_remove_noise_and_allignamd_extract_rir_and_build_db {
    room=$1
    config=$2
    record=$3
    echo "snr_based_delete_and_remove_noise_and_allignamd_extract_rir_and_build_db $room $config $record"  
    record_base=$(echo $record | sed -e 's/spkno-0/spkno/')
    
    find $room/$config/$record_base-* -name rec*.bz2 | grep -v '\.ir\.wav' | wc -l
    
    N=$(find $room/$config/$record_base-* -name rec*.bz2 | grep -v '\.ir\.wav' | wc -l )
    if  [ "$N" -le 0 ]
    then
        echo "No $room/$config/$record_base-*/rec*.bz2 exists "
        return
    fi

    for i in $room/$config/$record_base-*/*.wav.bz2
    do
        echo $i | grep '\.ir\.wav' >/dev/null
	if [ $? = 0 ]
	then
		echo "$i is an impulse response wav file, omitting this file."
		continue
	fi
        i=$(echo $i | sed -e 's/.bz2//')
        wavfile=$(basename $i)
        wavdir=$(dirname $i)


        if [ ! -f $wavdir/.$wavfile ]
        then
            bunzip2 $i.bz2     
            echo "Working (snr/noise_reduce/allign) on $i ..."
        
            python3 $SCRIPT_DIR/calculate_snr.py $i | grep SNR_NEGATIVE >/dev/null
            if [ $? == 0 ]
            then
                echo "mv $i $i.corrupted.snr beacuse of negative SNR"
                bzip2 $i
                mv $i.bz2 $i.bz2.corrupted_snr

	        #record=$(dirname $i | sed -e "s#$room/$config/#")
	        #mkdir -p $room/$config/trash/$record
	        #mv $i $room/$config/trash/$record
                #bzip2 $room/$config/trash/$record/$wavfile
            else
                 python3 $SCRIPT_DIR/noise_reducer.py $i
	         echo $i| grep -i song>/dev/null
	         if [ $? = 0 ]
	         then
	             python3 $SCRIPT_DIR/alligner.py $i $DATA_DIR/transmittedSongSignal.wav
                 else
	             python3 $SCRIPT_DIR/alligner.py $i $DATA_DIR/transmittedEssSignal.wav
    	         fi
    	         
    	         echo $i| grep -i ess>/dev/null
	         if [ $? = 0 ]
	         then
  		     python3 $SCRIPT_DIR/rir.py $i $DATA_DIR/transmittedEssSignal.wav
    	         fi
    	         
    	         insert_into_db $room $config $wavdir $wavfile
            fi
             
            bzip2 $i*
            touch $wavdir/.$wavfile
        else
            echo "Skipping $i which is already processed..."
        fi
    done
    

}

function get_field_value {
 recordfile=$1
 fieldname=$2
 returnValue=$(cat $recordfile|grep "$fieldname=" | cut -d\= -f2)
 if [ "$returnValue" = "" ]
 then
     returnValue="X"
 fi
 echo $returnValue
     
}

function insert_into_db {

 room=$1
 config=$2
 wavdir=$3
 wavfile=$4
 
 micNo=$(echo $wavfile| cut -d- -f2| cut -d. -f1)
 fields="timestamp speakerMotorIterationNo microphoneMotorIterationNo speakerMotorIterationDirection currentActiveSpeakerNo currentActiveSpeakerChannelNo physicalSpeakerNo microphoneStandInitialCoordinateX microphoneStandInitialCoordinateY microphoneStandInitialCoordinateZ speakerStandInitialCoordinateX speakerStandInitialCoordinateY speakerStandInitialCoordinateZ microphoneMotorPosition speakerMotorPosition temperatureAtMicrohponeStand humidityAtMicrohponeStand temperatureAtMSpeakerStand humidityAtSpeakerStand tempHumTimestamp speakerRelativeCoordinateX speakerRelativeCoordinateY speakerRelativeCoordinateZ microphoneStandAngle speakerStandAngle speakerAngleTheta speakerAnglePhi "



 headerline=""
 touch $room/$config/db.csv
 N=$(wc -l $room/$config/db.csv | awk '{print $1}')
 if [ "$N" = "0" ]
 then
	 headerline=$(echo $fields | tr ' ' ',')
	 headerline="$headerline,mic_RelativeCoordinateX,mic_RelativeCoordinateY,mic_RelativeCoordinateZ,mic_DirectionX,mic_DirectionY,mic_DirectionZ,mic_Theta,mic_Phi,filepath"
         echo "$headerline" >> $room/$config/db.csv

 fi

 dataline=""
 for fieldname in $(echo $fields)
 do
	 fieldvalue=$(get_field_value $wavdir/record.txt $fieldname)
	 if [ "$dataline" == "" ]
	 then
             dataline="$fieldvalue"
         else
             dataline="$dataline,$fieldvalue"
	 fi
 done
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_RelativeCoordinateX)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_RelativeCoordinateY)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_RelativeCoordinateZ)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_DirectionX)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_DirectionY)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_DirectionZ)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_Theta)
 dataline="$dataline,$fieldvalue"
 fieldvalue=$(get_field_value $wavdir/record.txt mic_${micNo}_Phi)
 dataline="$dataline,$fieldvalue"
 wavdir=$(echo $wavdir | sed -e 's#.*room-#room-#')
 dataline="$dataline,'$wavdir/$wavfile'"
 echo "$dataline" >> $room/$config/db.csv

}


(
for room in $(ls | grep room-)
do  
       	echo "## Room :  $room  "; 
        for config in $(ls $room| grep mic)
	do
                THERE_EXIST_MECHANICAL_PROBLEM=0
       	        echo "#### Config :  $config  "
                if [ ! -f $room/$config/.jpgclean ]
                then
                   for record in $(ls $room/$config| grep spkno-0)
	           do
	                if [ "$THERE_EXIST_MECHANICAL_PROBLEM" != "0" ]
	                then
	                   echo "There was a problem on step $THERE_EXIST_MECHANICAL_PROBLEM,  so we are moving the rest of the records to trash directory : $record"
	                   delete_record $room $config $record
	                else
       	                   echo "###### Record :  $record  "; 
                           eog $room/$config/$record/*.jpg
                           echo "Is the speaker/microphone mechanism moved appropirately?"
                           echo "-> Press n to delete this record and all the rest, Press Enter to continue without deleting this record"
                           read ans
                           if [ "$ans" = "n" ]
                           then
                            delete_record $room $config $record
                            THERE_EXIST_MECHANICAL_PROBLEM=$record
                           fi    
                        fi        
		  done
		  touch $room/$config/.jpgclean
	       fi  
	done
done

for room in $(ls | grep room-)
do  
       	echo "## Room :  $room  "; 
        for config in $(ls $room| grep mic)
	do
       	        echo "#### Config :  $config  "
                for record in $(ls $room/$config| grep spkno-0)
	        do
       	                   echo "###### Record :  $record  "; 
                           snr_based_delete_and_remove_noise_and_allignamd_extract_rir_and_build_db $room $config $record
		done
	done
done

###
### UPDATE DB.CSV THAT CONTAINS ALL RECORDS
###
TIMESTAMP=$(date '+%Y%m%d%H%M%S')
mv ess_db.csv ess_db.$TIMESTAMP.csv
mv song_db.csv song_db.$TIMESTAMP.csv

for room in $(ls | grep room-)
do  
       	echo "## Room :  $room  "; 
        for config in $(ls $room| grep mic)
	do
       	    echo "#### Config :  $config  "
	    grep wav $room/$config/db.csv| grep -i ess | sort | uniq | grep -v ir.wav >> ess_db.csv
	    grep wav $room/$config/db.csv| grep -i song | sort | uniq | grep -v ir.wav >> song_db.csv
	done
done


)2>&1 | tee /var/tmp/06.data.post.processing.log

























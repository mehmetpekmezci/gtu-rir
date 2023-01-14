#!/bin/bash

SCRIPT_DIR=$(realpath $(pwd))


DATA_DIR=$(realpath "../../../data/single-speaker/");
cd $DATA_DIR



function extract_rir {
    room=$1
    config=$2
    record=$3
    echo "snr_based_delete_and_remove_noise_and_allignamd_extract_rir_and_build_db $room $config $record"  
    record_base=$(echo $record | sed -e 's/spkno-0/spkno/')
    bunzip2 $room/$config/$record_base-*/receivedEssSignal-?.wav.bz2
    for i in $room/$config/$record_base-*/receivedEssSignal-?.wav
    do
        wavfile=$(basename $i)
        wavdir=$(dirname $i)
        echo "Working (snr/noise_reduce/allign) on $i ..."
        python3 $SCRIPT_DIR/rir.py $i $DATA_DIR/transmittedEssSignal.wav
    done
    bzip2 $room/$config/$record_base-*/*.wav
}

(
for room in $(ls | grep room-)
do  
       	echo "## Room :  $room  "; 
        for config in $(ls $room| grep mic)
	do
       	        echo "#### Config :  $config  "
                #for record in $(ls $room/$config| grep spkno-0)
                for record in $(ls $room/$config| grep spkno-0)
	        do
       	            echo "###### Record :  $record  "; 
                    extract_rir $room $config $record
	        done
	done
done

)2>&1 | tee /var/tmp/06.data.post.processing.log

























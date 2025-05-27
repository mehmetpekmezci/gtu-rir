#!/bin/bash

CURRENT_DIR=$(pwd)
GTURIR_DATA_DIR=$HOME/RIR_DATA/GTU-RIR-1.0/data/single-speaker/
RIR_GENERATOR="../rir-generators-4/MESH2IR/evaluate/single_record_generator.py"
ANALYSIS_DIR=$HOME/RIR_AXIS_ANALYSIS_DIR
mkdir -p $ANALYSIS_DIR

for i in $(cat selected_records.txt)
do   
	RECORD_NAME=$(echo $i | sed -e 's#/micx-\*/#-#' | sed -e 's#/receivedEssSignal#-micno#' | sed -e 's/.wav.ir.wav//')
	RECORD_SOURCE_DIR=$(dirname $GTURIR_DATA_DIR/$i)
	RECORD_DIR=$ANALYSIS_DIR/$RECORD_NAME
	rm -Rf $RECORD_DIR
	mkdir -p $RECORD_DIR
	cp $GTURIR_DATA_DIR/$i.bz2 $RECORD_DIR/real.rir.wav.bz2
	bunzip2 $RECORD_DIR/real.rir.wav.bz2
        RECORD_FILE=$RECORD_DIR/real.rir.wav 
        ROOM_ID=$(echo $RECORD_NAME | cut -d- -f1,2)
	MIC_ID=$(echo $RECORD_NAME| awk -F- '{print $NF}')
        cat $GTURIR_DATA_DIR/$ROOM_ID/properties/record.ini | grep '=' >  $RECORD_DIR/room_dims.txt
        grep "StandInitialCoordinate\|mic_${MIC_ID}_RelativeCoordinate\|speakerRelativeCoordinate" $RECORD_SOURCE_DIR/record.txt > $RECORD_DIR/record.txt
	perl -pi -e "s/mic_${MIC_ID}/mic/" $RECORD_DIR/record.txt
	echo "[all]" > $RECORD_DIR/properties.ini
	cat $RECORD_DIR/record.txt $RECORD_DIR/room_dims.txt >> $RECORD_DIR/properties.ini
	rm -f  $RECORD_DIR/record.txt $RECORD_DIR/room_dims.txt 
        cp ../rir-generators-4/EVALUATION_DATA-PREPARE-MESH2IR/ROOM_MESHES/GTURIR/$ROOM_ID-*.obj $RECORD_DIR/mesh.obj
        python3 generate_permutation.py $RECORD_DIR 
done





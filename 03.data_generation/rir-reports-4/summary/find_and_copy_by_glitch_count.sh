#!/bin/bash


if [ $# -lt 4 ]
then
	echo "Usage : $0 <DATA_DIR> <REPORT_DIR> <REF_METRIC_TYPE> <DATASET>"
	exit 1
fi

DATA_DIR=$1
REPORTS_DIR=$2
REF_METRIC_TYPE=$3 # MAX/MIN
DATASET=$4 # GTURIR
#REFERENCE_METRIC=MSE
REFERENCE_METRIC=GLITCH_COUNT #MSE

echo "$0 $DATA_DIR $REPORTS_DIR $REF_METRIC_TYPE $DATASET STARTED"

GTURIR_DATA_DIR=$DATA_DIR/GTU-RIR-1.0/data/single-speaker

if [ "$REF_METRIC_TYPE" = "MAX" ]
then
      # MX
      REF_METRIC_VALUE=$(cat $REPORTS_DIR/$DATASET/MESH2IR/*/*/$REFERENCE_METRIC.db.txt | grep -v 'e-'| cut -d'=' -f2| sort -g | tail -2 | head -1)
elif [ "$REF_METRIC_TYPE" = "MIN" ]
then
      # MIN
      REF_METRIC_VALUE=$(cat $REPORTS_DIR/$DATASET/MESH2IR/*/*/$REFERENCE_METRIC.db.txt | grep -v 'e-'| cut -d'=' -f2| sort -g | head -50| tail -1)
elif [ "$REF_METRIC_TYPE" = "AVG" ]
then
      # AVERAGE
      COUNT=$(cat $REPORTS_DIR/$DATASET/MESH2IR/*/*/$REFERENCE_METRIC.db.txt | grep -v 'e-'| cut -d'=' -f2| sort -g | wc -l)
      #COUNT=$( echo "scale=2 ; $COUNT / 2" | bc)
      COUNT=$(($COUNT/2))
      REF_METRIC_VALUE=$(cat $REPORTS_DIR/$DATASET/MESH2IR/*/*/$REFERENCE_METRIC.db.txt | grep -v 'e-'| cut -d'=' -f2| sort -g | head -$COUNT | tail -1)
fi 
mkdir -p $REPORTS_DIR/$DATASET/MESH2IR/fake/fake/
touch $REPORTS_DIR/$DATASET/MESH2IR/fake/fake/$REFERENCE_METRIC.db.txt

REF_METRIC_ROOM=$(basename $(dirname $(dirname $(grep =$REF_METRIC_VALUE $REPORTS_DIR/$DATASET/MESH2IR/*/*/$REFERENCE_METRIC.db.txt| head -1 | cut -d'=' -f1| cut -d: -f1))))
RELATED_RECORD=$(grep =$REF_METRIC_VALUE $REPORTS_DIR/$DATASET/MESH2IR/$REF_METRIC_ROOM/*/$REFERENCE_METRIC.db.txt| head -1 | cut -d'=' -f1| cut -d: -f2)
echo "ROOM=$REF_METRIC_ROOM"
echo "RECORD=$RELATED_RECORD"
   
SPEAKER_ITERATION=$(echo $RELATED_RECORD | cut -d- -f2)
MICROPHONE_ITERATION=$(echo $RELATED_RECORD | cut -d- -f4)
PHYSICAL_SPRAKER_NO=$(echo $RELATED_RECORD | cut -d- -f6)
MICROPHONE_NO=$(echo $RELATED_RECORD | cut -d- -f8)

echo "SPEAKER_ITERATION=$SPEAKER_ITERATION"
echo "MICROPHONE_ITERATION=$MICROPHONE_ITERATION"
echo "PHYSICAL_SPRAKER_NO=$PHYSICAL_SPRAKER_NO"
echo "MICROPHONE_NO=$MICROPHONE_NO"

SUMMARY_DIR=$REPORTS_DIR/summary/$REF_METRIC_TYPE-$REFERENCE_METRIC-$DATASET

echo SUMMARY_DIR=$SUMMARY_DIR

rm -Rf $SUMMARY_DIR
mkdir -p $SUMMARY_DIR

if [ "$DATASET" == "GTURIR" ]
then
	   cp $GTURIR_DATA_DIR/$REF_METRIC_ROOM/micx*/micstep-$MICROPHONE_ITERATION-spkstep-$SPEAKER_ITERATION-spkno-$PHYSICAL_SPRAKER_NO/receivedSongSignal-$MICROPHONE_NO.wav.bz2 $SUMMARY_DIR/real.song.$REF_METRIC_ROOM-micstep-$MICROPHONE_ITERATION-spkstep-$SPEAKER_ITERATION-spkno-$PHYSICAL_SPRAKER_NO-micno-$MICROPHONE_NO.wav.bz2
	   bunzip2 $SUMMARY_DIR/real.song.$REF_METRIC_ROOM-micstep-$MICROPHONE_ITERATION-spkstep-$SPEAKER_ITERATION-spkno-$PHYSICAL_SPRAKER_NO-micno-$MICROPHONE_NO.wav.bz2
	   cp $GTURIR_DATA_DIR/$REF_METRIC_ROOM/micx*/micstep-$MICROPHONE_ITERATION-spkstep-$SPEAKER_ITERATION-spkno-$PHYSICAL_SPRAKER_NO/receivedEssSignal-$MICROPHONE_NO.wav.ir.wav.bz2 $SUMMARY_DIR/real.rir.$REF_METRIC_ROOM-micstep-$MICROPHONE_ITERATION-spkstep-$SPEAKER_ITERATION-spkno-$PHYSICAL_SPRAKER_NO-micno-$MICROPHONE_NO.wav.bz2
	   bunzip2 $SUMMARY_DIR/real.rir.$REF_METRIC_ROOM-micstep-$MICROPHONE_ITERATION-spkstep-$SPEAKER_ITERATION-spkno-$PHYSICAL_SPRAKER_NO-micno-$MICROPHONE_NO.wav.bz2
fi
REAL_RIR=$(find $SUMMARY_DIR -name 'real.rir.*.wav' | grep -v reverbed| head -1)

echo "REAL_RIR=$REAL_RIR"

python3 convolve.py $HOME/RIR_REPORT/transmittedSongSignal.wav $REAL_RIR

### generating coherence plots Model1-Generated-RIR vs. Real-RIR
for GENERATORTYPE in MESH2IR MESHTAE
do
           #for i in $REPORTS_DIR/$DATASET/$GENERATORTYPE*
           for i in $REPORTS_DIR/$DATASET/$GENERATORTYPE
           do
	       GENERATOR=$(basename $i)
	       mkdir -p $SUMMARY_DIR/$GENERATOR
	       if [ -d $i/summary ]
	       then
	        echo cp $i/$REF_METRIC_ROOM/*/$RELATED_RECORD.* $SUMMARY_DIR/$GENERATOR
	        cp $i/$REF_METRIC_ROOM/*/$RELATED_RECORD.* $SUMMARY_DIR/$GENERATOR
		echo ./coherence_plot.sh $DATA_DIR $REPORTS_DIR $REF_METRIC_TYPE $DATASET $RELATED_RECORD $REAL_RIR $GENERATOR $i $REFERENCE_METRIC
		./coherence_plot.sh $DATA_DIR $REPORTS_DIR $REF_METRIC_TYPE $DATASET $RELATED_RECORD $REAL_RIR $GENERATOR $i $REFERENCE_METRIC &
	       else
		 echo "$i/summary not found !!!"
	       fi
           done
done

### generating coherence plots Model1-Generated-RIR vs. Model2-Generated-RIR  
./coherence_plot_model1_vs_model2.sh $DATA_DIR $REPORTS_DIR $REF_METRIC_TYPE $DATASET $RELATED_RECORD MESH2IR MESHTAE $REFERENCE_METRIC &

wait
echo "$0 $DATA_DIR $REPORTS_DIR $REF_METRIC_TYPE $DATASET FINISHED"

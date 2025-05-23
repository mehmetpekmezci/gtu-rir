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
RELATED_RECORD=$5 
REAL_RIR=$6 
GENERATOR=$7
i=$8
REFERENCE_METRIC=$9

GTURIR_DATA_DIR=$DATA_DIR/GTU-RIR-1.0/data/single-speaker
SUMMARY_DIR=$REPORTS_DIR/summary/$REF_METRIC_TYPE-$REFERENCE_METRIC-$DATASET

echo "$0 : SUMMARY_DIR=$SUMMARY_DIR $RELATED_RECORD $GENERATOR STARTED"

echo "i/summary = $i/summary"

if [ -d $i/summary ]
then
                echo python3 coherence_plot.py $REAL_RIR $SUMMARY_DIR/$GENERATOR/$RELATED_RECORD*.wav
                python3 coherence_plot.py $REAL_RIR $SUMMARY_DIR/$GENERATOR/$RELATED_RECORD*.wav
		echo python3 convolve.py $HOME/RIR_REPORT/transmittedSongSignal.wav $SUMMARY_DIR/$GENERATOR/$RELATED_RECORD*.wav
		python3 convolve.py $HOME/RIR_REPORT/transmittedSongSignal.wav $SUMMARY_DIR/$GENERATOR/$RELATED_RECORD*.wav
fi

echo "$0 : SUMMARY_DIR=$SUMMARY_DIR $RELATED_RECORD $GENERATOR FINISHED"

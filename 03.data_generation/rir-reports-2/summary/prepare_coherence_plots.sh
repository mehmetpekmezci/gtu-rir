#!/bin/bash


if [ $# -lt 2 ]
then
	echo "Usage : $0 <DATA_DIR> <REPORT_DIR>"
	exit 1
fi

DATA_DIR=$1
REPORTS_DIR=$2
REFERENCE_METRIC=MSE

GTURIR_DATA_DIR=$DATA_DIR/GTU-RIR-1.0/data/single-speaker
BUT_RIR_DATA_DIR=$DATA_DIR/BUTReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only

./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR MAX BUTReverbDB &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR MIN BUTReverbDB &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR AVG BUTReverbDB &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR MAX GTURIR &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR MIN GTURIR &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR AVG GTURIR &

wait

./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR MAX BUTReverbDB &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR MIN BUTReverbDB &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR AVG BUTReverbDB &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR MAX GTURIR &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR MIN GTURIR &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR AVG GTURIR &

wait

echo "$0 is FINISHED"

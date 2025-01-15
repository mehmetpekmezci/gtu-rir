#!/bin/bash


if [ $# -lt 2 ]
then
	echo "Usage : $0 <DATA_DIR> <REPORT_DIR>"
	exit 1
fi

DATA_DIR=$1
REPORTS_DIR=$2

GTURIR_DATA_DIR=$DATA_DIR/GTU-RIR-1.0/data/single-speaker

./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR MAX GTURIR &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR MIN GTURIR &
./find_and_copy_by_glitch_count.sh $DATA_DIR $REPORTS_DIR AVG GTURIR &

wait

./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR MAX &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR MIN &
./coherence_plot_group.sh $DATA_DIR $REPORTS_DIR AVG &

wait

echo "$0 is FINISHED"

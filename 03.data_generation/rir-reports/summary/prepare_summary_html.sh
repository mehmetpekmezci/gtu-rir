#!/bin/bash

CURRENT_DIR=$(pwd)

DEFAULT_MAIN_REPORT_DIR=$HOME/RIR_REPORT/

echo "<html><head></head><body><table border='1'>" > $DEFAULT_MAIN_REPORT_DIR/summary.html
echo "<tr><th>DATASET_NAME</th><th>GAN_MODEL</th><th>TRAINING_LOSS_FN</th><th>MEAN_MSE</th><th>MEAN_SSIM</th><th>MEAN_MFCC_MSE</th><th>MEAN_MFCC_SSIM</th><th>MEAN_MFCC_CROSS_ENTROPY</th><th>MEAN_GLITCH_COUNT</th></tr>">> $DEFAULT_MAIN_REPORT_DIR/summary.html

cd $DEFAULT_MAIN_REPORT_DIR

for DATASET_NAME in GTURIR BUTReverbDB
do
     REPORT_DIR=$DEFAULT_MAIN_REPORT_DIR/$DATASET_NAME
     cd $REPORT_DIR
     for GANMODEL in FASTRIR MESH2IR
     do
	if [ "$BGCOLOR" = "BBBBBB" ]
	then 
		BGCOLOR="EEEEEE"
	else
		BGCOLOR="BBBBBB"
	fi
	TRAINING_LOSS_FNS=$(ls -d MESH2IR-*| sed -e "s#MESH2IR-##g")
	for TRAINING_LOSS_FN in $TRAINING_LOSS_FNS
	do
	   MEAN_MSE=""
	   MEAN_SSIM=""
	   MEAN_MFCC_MSE=""
	   MEAN_MFCC_SSIM=""
	   MEAN_MFCC_CROSS_ENTROPY=""
	   MEAN_GLITCH_COUNT=""

	   if [ -f $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt ]
	   then
	       MEAN_MSE=$(cat $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt| grep 'MEAN_MSE=' | cut -d\= -f2)
	       MEAN_SSIM=$(cat $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt| grep 'MEAN_SSIM=' | cut -d\= -f2)
	       MEAN_MFCC_MSE=$(cat $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt| grep 'MEAN_MFCC_MSE=' | cut -d\= -f2)
	       MEAN_MFCC_SSIM=$(cat $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt| grep 'MEAN_MFCC_SSIM=' | cut -d\= -f2)
	       MEAN_MFCC_CROSS_ENTROPY=$(cat $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt| grep 'MEAN_MFCC_CROSS_ENTROPY=' | cut -d\= -f2)
	       MEAN_GLITCH_COUNT=$(cat $GANMODEL-$TRAINING_LOSS_FN/summary/summary.db.txt| grep 'MEAN_GLITCH_COUNT=' | cut -d\= -f2)
	   fi
	   echo "<tr bgcolor='#$BGCOLOR'><td>$DATASET_NAME</td><td>$GANMODEL</td><td>$TRAINING_LOSS_FN</td><td>$MEAN_MSE</td><td>$MEAN_SSIM</td><td>$MEAN_MFCC_MSE</td><td>$MEAN_MFCC_SSIM</td><td>$MEAN_MFCC_CROSS_ENTROPY</td><td>$MEAN_GLITCH_COUNT</td></tr>" >> $DEFAULT_MAIN_REPORT_DIR/summary.html
        done
     done
done
echo "</table></body></html>" >> $DEFAULT_MAIN_REPORT_DIR/summary.html


cp $HOME/RIR_DATA/GTU-RIR-1.0/data/single-speaker/transmittedSongSignal.wav.bz2 $HOME/RIR_REPORT
bunzip2 $HOME/RIR_REPORT/transmittedSongSignal.wav.bz2

cd $CURRENT_DIR

./prepare_coherence_plots.sh $HOME/RIR_DATA $HOME/RIR_REPORT

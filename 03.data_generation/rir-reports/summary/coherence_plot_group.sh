#!/bin/bash


if [ $# -lt 4 ]
then
	echo "Usage : $0 <DATA_DIR> <REPORT_DIR> <REF_METRIC_TYPE> <DATASET> "
	exit 1
fi

DATA_DIR=$1
REPORTS_DIR=$2
REF_METRIC_TYPE=$3 # MAX/MIN
DATASET=$4 # BUT_REVERBDB/GTURIR
REFERENCE_METRIC=GLITCH_COUNT #MSE

GTURIR_DATA_DIR=$DATA_DIR/GTU-RIR-1.0/data/single-speaker
BUT_RIR_DATA_DIR=$DATA_DIR/BUTReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only

SUMMARY_DIR=$REPORTS_DIR/summary/$REF_METRIC_TYPE-$REFERENCE_METRIC-$DATASET

#FASTRIR-MSE          --  FASTRIR-SSIM        --   FASTRIR-SSIM_PLUS_MSE
#FASTRIR-MFCC-MSE     --  FASTRIR-MFCC-SSIM   --   FASTRIR-MFCC-SSIM-AND-MSE-WEIGHTED
#MESH2IR-MSE           --  MESH2IR-SSIM        --   MESH2IR-SSIM_PLUS_MSE
#MESH2IR-MFCC-MSE     --  MESH2IR-MFCC-SSIM   --   MESH2IR-MFCC-SSIM-AND-MSE-WEIGHTED

PLOT_GROUP_PARAM=""
PLOT_GROUP_PARAM_2=""
for model in FASTRIR MESH2IR
do
      for metric in MSE SSIM SSIM_PLUS_MSE MFCC-MSE MFCC-SSIM MFCC-SSIM-AND-MSE-WEIGHTED 
      do
	     if [ ! -d $SUMMARY_DIR/$model-$metric ]
	     then
		     mkdir -p $SUMMARY_DIR/$model-$metric
	     fi
	     if [ ! -f $SUMMARY_DIR/$model-$metric/*.new.png ]
	     then
		     convert -size 32x32 xc:white $SUMMARY_DIR/$model-$metric/white.new.png
	     fi
	     if [ ! -f $SUMMARY_DIR/$model-$metric/*.new_real_front.png ]
	     then
		     convert -size 32x32 xc:white $SUMMARY_DIR/$model-$metric/white.new_real_front.png
	     fi
	     PLOT_GROUP_PARAM="$PLOT_GROUP_PARAM $SUMMARY_DIR/$model-$metric/*.new.png"
	     PLOT_GROUP_PARAM_2="$PLOT_GROUP_PARAM_2 $SUMMARY_DIR/$model-$metric/*.new_real_front.png"
      done
done
echo python3 coherence_plot_group.py $SUMMARY_DIR "GENERATED_ON_TOP" $PLOT_GROUP_PARAM
python3 coherence_plot_group.py $SUMMARY_DIR "GENERATED_ON_TOP" $PLOT_GROUP_PARAM
echo python3 coherence_plot_group.py $SUMMARY_DIR "REAL_ON_TOP" $PLOT_GROUP_PARAM_2
python3 coherence_plot_group.py $SUMMARY_DIR "REAL_ON_TOP" $PLOT_GROUP_PARAM_2



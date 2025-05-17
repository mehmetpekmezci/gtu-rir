#!/bin/bash


if [ $# -lt 3 ]
then
	echo "Usage : $0 <DATA_DIR> <REPORT_DIR> <REF_METRIC_TYPE> "
	exit 1
fi

DATA_DIR=$1
REPORTS_DIR=$2
REF_METRIC_TYPE=$3 # MAX/MIN/AVG


SUMMARY_DIR=$REPORTS_DIR/summary
SUMMARY_PLOT_GROUP_DIR=$REPORTS_DIR/summary/$REF_METRIC_TYPE

if [ ! -d $SUMMARY_PLOT_GROUP_DIR ]
then
     mkdir -p $SUMMARY_PLOT_GROUP_DIR
fi


for dataset in GTURIR BUTReverbDB
do
      PLOT_GROUP_PARAM=""
      PLOT_GROUP_PARAM_2=""
      for model in MESHTAE MESH2IR
      do
	     if [ ! -f $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/$model/*.new.png ]
	     then
		     convert -size 32x32 xc:white $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/$model/white.new.png
	     fi
	     if [ ! -f $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/$model/*.new_real_front.png ]
	     then
		     convert -size 32x32 xc:white $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/$model/white.new_real_front.png
	     fi
	     PLOT_GROUP_PARAM="$PLOT_GROUP_PARAM $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/$model/*.new.png"
	     PLOT_GROUP_PARAM_2="$PLOT_GROUP_PARAM_2 $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/$model/*.new_real_front.png"
      done
      echo python3 coherence_plot_group.py $SUMMARY_PLOT_GROUP_DIR "GENERATED_ON_TOP" $dataset $PLOT_GROUP_PARAM $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/MESHTAE/*.model_comparison.png
      python3 coherence_plot_group.py $SUMMARY_PLOT_GROUP_DIR "GENERATED_ON_TOP" $dataset $PLOT_GROUP_PARAM $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/MESHTAE/*.model_comparison.png
      echo python3 coherence_plot_group.py $SUMMARY_PLOT_GROUP_DIR "REAL_ON_TOP" $dataset $PLOT_GROUP_PARAM_2 $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/MESHTAE/*.model_comparison_real_front.png
      python3 coherence_plot_group.py $SUMMARY_PLOT_GROUP_DIR "REAL_ON_TOP" $dataset $PLOT_GROUP_PARAM_2 $SUMMARY_DIR/$REF_METRIC_TYPE-*-$dataset/MESHTAE/*.model_comparison_real_front.png
done


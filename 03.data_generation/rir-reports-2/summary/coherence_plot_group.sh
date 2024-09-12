#!/bin/bash


if [ $# -lt 3 ]
then
	echo "Usage : $0 <DATA_DIR> <REPORT_DIR> <REF_METRIC_TYPE> "
	exit 1
fi

DATA_DIR=$1
REPORTS_DIR=$2
REF_METRIC_TYPE=$3 # MAX/MIN

SUMMARY_DIR=$REPORTS_DIR/summary/$REF_METRIC_TYPE-$REFERENCE_METRIC

PLOT_GROUP_PARAM=""
PLOT_GROUP_PARAM_2=""
for model in MESHTAE MESH2IR
do
      for dataset in GTURIR BUTReverbDB
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



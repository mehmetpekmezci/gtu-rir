#!/bin/bash

CLEAN_ALL="TRUE"
CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA
DEFAULT_MAIN_REPORT_DIR=$HOME/RIR_REPORT
REPORT_DIR=$DEFAULT_MAIN_REPORT_DIR/GTURIR/

if [ "$GTURIR_DATA" = "" ]
then
        export GTURIR_DATA=$DEFAULT_MAIN_DATA_DIR/GTU-RIR-1.0/data/single-speaker/
        echo "GTURIR_DATA=$GTURIR_DATA"
fi


if [ "$GANMODELS" = "" ]
then
        GANMODELS="MESHTAE MESH2IR"
fi

if [ "$RIR_GENERATORS_DIR" = "" ]
then
        RIR_GENERATORS_DIR="../../rir-generators-4"
fi

for GANMODEL in $GANMODELS
do
for i in $RIR_GENERATORS_DIR/$GANMODEL*
do
MODEL=$(basename $i) 
echo grep "#$MODEL#" ignore_models.conf
grep "#$MODEL#" ignore_models.conf
if [ $? = 0 ]
then
	echo "IGNORING $MODEL .."
	continue
fi
if [ -f $REPORT_DIR/$MODEL/summary/summary.db.txt ]
then
        echo "Summary report is already generated, skipping $MODEL"
    else
        echo "$MODEL STARTED :"
	date
	cd $CURRENT_DIR/$i
	if [ -f run_eval.sh ]
	then
	   if [ ! -f $REPORT_DIR/$MODEL/.successfulyGenerated ]
	   then
               echo "Starting to generate RIR signals using $MODEL model"
	       echo ./run_eval.sh $GTURIR_DATA $REPORT_DIR/$MODEL GTURIR
	       ./run_eval.sh $GTURIR_DATA $REPORT_DIR/$MODEL GTURIR
	       if [ $? = 0 ]
	       then
		   touch $REPORT_DIR/$MODEL/.successfulyGenerated
	       else
	           echo "Colud not generate RIR signals using $MODEL model, exiting ...."
	       fi
               echo "Finished to generate RIR signals using $MODEL model"
	   fi
          
	   cd $CURRENT_DIR


	   if [ -f $REPORT_DIR/$MODEL/.successfulyGenerated ]
           then
	       echo "Starting to generate report data with $MODEL model"
	       if [ "$CLEAN_ALL" = "TRUE" ]
	       then
	            find $REPORT_DIR/$MODEL/ -name *.db.txt| xargs rm -f
	            find $REPORT_DIR/$MODEL/ -name *.wavesAndSpectrogramsGenerated| xargs rm -f
	       fi
	       for roomId in $(echo room-207 room-208 room-conferrence01 room-sport01 room-sport02 room-z02 room-z04 room-z06 room-z10 room-z11 room-z23)
	       do
		   echo $roomId
                   python3 mainData.py $GTURIR_DATA $REPORT_DIR/$MODEL $roomId 
	       done
               echo "Finished to generate report data for $MODEL model"
	       echo "Starting to generate report table with $MODEL model"
               python3 mainReport.py $GTURIR_DATA $REPORT_DIR/$MODEL 
               echo "Finished to generate report table for $MODEL model"
	   fi

	fi
        echo "$MODEL ENDED :"
	date
fi
done
done





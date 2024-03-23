#!/bin/bash

./prepare_data.sh


#CLEAN_ALL="TRUE"
CLEAN_ALL="FALSE"
MAX_NUMBER_OF_PARALLEL_PROCESSES=4

CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA
DEFAULT_MAIN_REPORT_DIR=$HOME/RIR_REPORT
REPORT_DIR=$DEFAULT_MAIN_REPORT_DIR/BUTReverbDB/

if [ "$BUTReverbDB_DATA" = "" ]
then
        export BUTReverbDB_DATA=$DEFAULT_MAIN_DATA_DIR/BUTReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only
        echo "BUTReverbDB_DATA=$BUTReverbDB_DATA"
fi


if [ ! -f $BUTReverbDB_DATA/RIR.fast_rir_inputs.pickle ]
then	
     echo "$BUTReverbDB_DATA/RIR.fast_rir_inputs.pickle does not exist , generating ..."
     python3 generate_fast_rir_inputs_pickle.py $BUTReverbDB_DATA

fi


for GANMODEL in FASTRIR MESH2IR
do
for i in ../../rir-generators/$GANMODEL-*
do
MODEL=$(basename $i) 
echo "tail -f log.$MODEL"
(
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
	       if [ "$GANMODEL" = "FASTRIR" ]
               then
                    echo ./run_eval.sh $BUTReverbDB_DATA/RIR.fast_rir_inputs.pickle $REPORT_DIR/$MODEL
                    ./run_eval.sh $BUTReverbDB_DATA/RIR.fast_rir_inputs.pickle $REPORT_DIR/$MODEL
               elif [ "$GANMODEL" = "MESH2IR" ]
               then
                    echo ./run_eval.sh $BUTReverbDB_DATA $REPORT_DIR/$MODEL BUTReverbDB
                    ./run_eval.sh $BUTReverbDB_DATA $REPORT_DIR/$MODEL BUTReverbDB
               else
                    echo "GAN MODEL $GANMODEL NOT FOUND !"
                    exit 1
               fi

	       if [ $? = 0 ]
	       then
		   touch $REPORT_DIR/$MODEL/.successfulyGenerated
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
	       for roomId in $(echo Hotel_SkalskyDvur_ConferenceRoom2 Hotel_SkalskyDvur_Room112 VUT_FIT_C236 VUT_FIT_D105 VUT_FIT_E112 VUT_FIT_L207 VUT_FIT_L212 VUT_FIT_L227 VUT_FIT_Q301)
	       do
		   N=$(ps -ef | grep python3 | grep mainData.py | wc -l)
		   if [ $N -ge $MAX_NUMBER_OF_PARALLEL_PROCESSES ]
		   then
			   echo "waiting , more than 4 parallel processes ...."
			   wait
		   fi
		   echo $roomId
                   python3 mainData.py $BUTReverbDB_DATA $REPORT_DIR/$MODEL $roomId &
	       done
	       wait
               echo "Finished to generate report data for $MODEL model"

	       echo "Starting to generate report table with $MODEL model"
               python3 mainReport.py $BUTReverbDB_DATA $REPORT_DIR/$MODEL 
               echo "Finished to generate report table for $MODEL model"

	   fi
	fi
        echo "$MODEL ENDED :"
	date
    fi
)&> log.$MODEL
done
done




#!/bin/bash

MAX_NUMBER_OF_PARALLEL_PROCESSES=30
NUMBER_OF_SAMPLES=400

NUMBER_OF_PARALLEL_PROCESS=0

if [ ! -f $HOME/RIR_DATA/APARTMENT_MESH/.objfiles_are_generated ]
then

for((width=10;width<25;width++))
do
   for((depth=10;depth<25;depth++))
   do
      for((height=0;height<30;height++))
      do
        WIDTH=$width
        DEPTH=$depth
        HEIGHT=$(echo "scale=2; 3+$height/10" | bc);
        DATA_DIR=$HOME/RIR_DATA/APARTMENT_MESH/WIDTH_${WIDTH}xDEPTH_${DEPTH}xHEIGHT_${HEIGHT}
	if [ ! -d $DATA_DIR ]
	then
           echo "WIDTH_${WIDTH}xDEPTH_${DEPTH}xHEIGHT_${HEIGHT}   started ..."
           date
           mkdir -p $DATA_DIR
           NUMBER_OF_PARALLEL_PROCESS=$((NUMBER_OF_PARALLEL_PROCESS+1))
           (
              python3 main.py $WIDTH $DEPTH $HEIGHT $NUMBER_OF_SAMPLES $DATA_DIR 
              #gzip $DATA_DIR/*.obj
            ) &  
           if [ $NUMBER_OF_PARALLEL_PROCESS -gt $MAX_NUMBER_OF_PARALLEL_PROCESSES ]
           then
             echo "WAITING FOR PARALLEL PROCESSES : $NUMBER_OF_PARALLEL_PROCESS"
             wait
             NUMBER_OF_PARALLEL_PROCESS=0
           fi
	fi 

      done
  done
done

touch $HOME/RIR_DATA/APARTMENT_MESH/.objfiles_are_generated

fi


#NUMBER_OF_PARALLEL_PROCESS=0
#
#for objfile in $(find $HOME/RIR_DATA/APARTMENT_MESH/ -name '*.obj')
#do
#    NUMBER_OF_PARALLEL_PROCESS=$((NUMBER_OF_PARALLEL_PROCESS+1))
#    (
#      python3 ../../rir-generators/TRAINING_DATA-PREPARE-MESH2IR/graph_generator.py $objfile
#    ) &  
#    if [ $NUMBER_OF_PARALLEL_PROCESS -gt $MAX_NUMBER_OF_PARALLEL_PROCESSES ]
#    then
#     echo "WAITING FOR PARALLEL PROCESSES : $NUMBER_OF_PARALLEL_PROCESS"
#     wait
#     NUMBER_OF_PARALLEL_PROCESS=0
#    fi
#done

for((width=10;width<25;width++))
do
python3 graph_generator.py $HOME/RIR_DATA/APARTMENT_MESH $width &
done
             

             


#!/bin/bash

CURRENT_DIR=$(pwd)

DEFAULT_MAIN_DATA_DIR=$HOME/RIR_DATA

if [ "$FAST_RIR_TRAINING_DATA" = "" ]
then
	echo "FAST_RIR_TRAINING_DATA env. var. is not defined"
        export FAST_RIR_TRAINING_DATA=$DEFAULT_MAIN_DATA_DIR/FAST_RIR_TRAINING_DATA
        echo "FAST_RIR_TRAINING_DATA=$FAST_RIR_TRAINING_DATA"
fi

if [ ! -d "$FAST_RIR_TRAINING_DATA/data/Medium_Room" ]
then
	echo "$FAST_RIR_TRAINING_DATA does not exist ..."
        mkdir -p $FAST_RIR_TRAINING_DATA
        cd $FAST_RIR_TRAINING_DATA
        #gdown "https://drive.google.com/uc?id=17NF1MVtXaWe9zhqWJqmG5tFUZb_9X0M5"
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17NF1MVtXaWe9zhqWJqmG5tFUZb_9X0M5" -O data.zip && rm -rf /tmp/cookies.txt
        unzip data.zip	
else
	echo "$FAST_RIR_TRAINING_DATA/data/Medium_Room training data directory already exists"
fi


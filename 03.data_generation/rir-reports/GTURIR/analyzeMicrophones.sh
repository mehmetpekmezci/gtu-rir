#!/bin/bash

if [ ! -f ~/RIR_DATA/GTU-RIR/RIR.pickle.dat ]
then
	echo "Could not find data file !"
	exit 1
fi

if [ ! -f ~/RIR_DATA/GTU-RIR/RIR.pickle.dat.room-z23 ]
then
    python3 splitDataIntoRooms.py ~/RIR_DATA/GTU-RIR/RIR.pickle.dat
fi


python3 microphoneAnalysis.py ~/RIR_DATA/GTU-RIR/RIR.pickle.dat



#!/bin/bash

DATA_DIR=$HOME/RIR_DATA/GTURIR-ROOM-MESH
mkdir -p $DATA_DIR
python3 main.py $DATA_DIR 
python3 graph_generator.py $DATA_DIR 


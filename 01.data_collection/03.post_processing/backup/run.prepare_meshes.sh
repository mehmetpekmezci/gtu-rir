#!/bin/bash
exit 0
#sudo apt-get install libmpfr-dev libgmp-dev libboost-all-dev
#sudo pip3 install pymesh2
#sudo apt-get install python-dev python3-dev


DATA_DIR=$(realpath "../../../data/single-speaker/");
cd $DATA_DIR

for objFile in $DATA_DIR/room-*/properties/*.obj
do
     echo $objFile | grep normalized_2000_to_faces >/dev/null
     if [ $? != 0 ]
     then
          echo python3 mesh_simplification.py $objFile
          python3 mesh_simplification.py $objFile
     fi
done

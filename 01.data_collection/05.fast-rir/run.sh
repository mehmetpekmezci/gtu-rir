#!/bin/bash
rm -f log.old
mv log log.old
(
# sudo pip3 install tf-image



CURRENT_DIR=$(pwd)

FAST_RIR_DIR=$HOME/workspace-python/FAST-RIR

if [ ! -d $FAST_RIR_DIR ]
then
    
    mkdir -p  $HOME/workspace-python
    cd  $HOME/workspace-python
    git clone https://github.com/anton-jeran/FAST-RIR
    cd FAST-RIR
    cp -dpr $CURRENT_DIR/FAST-RIR-extras/* .
    source download_generate.sh
    source download_data.sh
    chmod +x *.sh
    cd $CURRENT_DIR    
fi
    
    

   
    
      

python3 main.py $FAST_RIR_DIR
)>&log &
tail -f log


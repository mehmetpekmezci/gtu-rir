#!/bin/bash

which ticcmd 
if [ $? != 0 ]
then
     echo "install ticcmd from ../../driver/pololu-tic-1.8.1-linux-x86.tar.xz if not already installed"
     exit 1
fi

N=$(ticcmd --list| wc -l )

if [ "$N" -lt 2 -a "$NO_CHECK" != "1" ]
then
   echo "All (2) devices are not connected !"
   ticcmd --list
   exit 1
fi   


./motor.reset.sh T500
./motor.reset.sh T825

STEP=10


for((i=0;$i<$STEP;i=i+1))
do
echo "STEP FORWARD : $i"
./motor.iterate.sh T500 1 $i TEST_MODE_NONE
./motor.iterate.sh T825 1 $i TEST_MODE_NONE
sleep 5
done


for((i=0;$i<$STEP;i=i+1))
do
echo "STEP AFTWARD : $i"
./motor.iterate.sh T500 0 $i TEST_MODE_NONE
./motor.iterate.sh T825 0 $i TEST_MODE_NONE
sleep 5
done


DEVICE=$(ticcmd --list|grep T500| cut -d, -f1)
ticcmd -d $DEVICE --deenergize
ticcmd -d $DEVICE --reset

DEVICE=$(ticcmd --list|grep T825| cut -d, -f1)
ticcmd -d $DEVICE --deenergize
ticcmd -d $DEVICE --reset




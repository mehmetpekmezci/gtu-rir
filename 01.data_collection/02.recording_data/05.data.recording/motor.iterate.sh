#!/bin/bash

if [ $# -lt 4 ]
then
    echo "Usage: $0 <T500/T825> <1/0> <STEP_NO> <TEST_MODE>"
    exit 1
fi

DEVICE_NAME=$1
DIRECTION=$2 ## 1 pozitif, 0 negatif
STEP=$3
TEST_MODE=$4

STEP=$((STEP+1))

TARGET_END_POSITION=$((STEP*20))

if [ "$DIRECTION" == 0 ]
then
	TARGET_END_POSITION=$((200-STEP*20))
fi

if [ "$TEST_MODE" != "TEST_MODE_NONE" ]
then
	echo $TARGET_END_POSITION > /tmp/dummy.$DEVICE_NAME.position.txt 
	echo "$DEVICE_NAME $DIRECTION $STEP $TEST_MODE $TARGET_END_POSITION">> /tmp/dummy.motor.history.txt 
	exit 0
fi

DEVICE=$(ticcmd --list|grep $DEVICE_NAME| cut -d, -f1)

if [ "$DEVICE" == "" ]
then
    echo "$1 device not found !!"
    exit 1
fi



ITERATION_DURATION=2

VELOCITY=100000

if [ "$DIRECTION" == 0 ]
then
	VELOCITY=-100000
fi


START_POSITION=$(ticcmd -d $DEVICE -s --full| grep position| grep -i current | cut -d: -f2 | awk '{print $1}')


echo "$1 Move Command started"
ticcmd -d $DEVICE --resume --exit-safe-start --velocity $VELOCITY &
echo "$1 Move Command given , now sleeping $ITERATION_DURATION"
sleep $ITERATION_DURATION
echo "$1 Halt and hold"
ticcmd -d $DEVICE --halt-and-hold
echo "$1 Halt and hold ended"

CURRENT_POSITION=$(ticcmd -d $DEVICE -s --full| grep position| grep -i current | cut -d: -f2 | awk '{print $1}')

echo "$DEVICE.CURRENT_POSITION=$CURRENT_POSITION"
echo "$DEVICE.TARGET_END_POSITION=$TARGET_END_POSITION"

#
#if [ "$DIRECTION" == 1 -a "$CURRENT_POSITION" -lt "$TARGET_END_POSITION" ]
#then
#     DIFF=$((TARGET_END_POSITION-CURRENT_POSITION))
#     VELOCITY=$((DIFF*10000))
#     ticcmd -d $DEVICE --resume --exit-safe-start --velocity $VELOCITY &
#     sleep 1
#     ticcmd -d $DEVICE --halt-and-hold
#     sleep 1
#     ticcmd -d $DEVICE --deenergize
#
#elif [ "$DIRECTION" == 0 -a "$CURRENT_POSITION" -lt "$TARGET_END_POSITION" ]
#then
#     DIFF=$((CURRENT_POSITION-TARGET_END_POSITION))
#     VELOCITY=$((DIFF*-10000))
#     ticcmd -d $DEVICE --resume --exit-safe-start --velocity $VELOCITY  &
#     sleep 1
#     ticcmd -d $DEVICE --halt-and-hold
#     sleep 1
#     ticcmd -d $DEVICE --deenergize
#fi
#
#
#CURRENT_POSITION=$(ticcmd -d $DEVICE -s --full| grep position| grep -i current | cut -d: -f2 | awk '{print $1}')
#
#echo "2.CURRENT_POSITION=$CURRENT_POSITION"
#echo "TARGET_END_POSITION=$TARGET_END_POSITION"
#
#
#ticcmd -d $DEVICE --resume --exit-safe-start --position $TARGET_END_POSITION  &
#ticcmd -d $DEVICE --halt-and-hold
sleep 1
ticcmd -d $DEVICE --deenergize

#CURRENT_POSITION=$(ticcmd -d $DEVICE -s --full| grep position| grep -i current | cut -d: -f2 | awk '{print $1}')
#echo "3.CURRENT_POSITION=$CURRENT_POSITION"
#echo "TARGET_END_POSITION=$TARGET_END_POSITION"

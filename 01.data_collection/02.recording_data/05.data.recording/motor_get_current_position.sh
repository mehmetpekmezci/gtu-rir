#!/bin/bash

if [ $# -lt 2 ]
then
    echo "Usage: $0 <T500/T825> <TEST_MODE>"
    exit 1
fi

if [ "$2" != "TEST_MODE_NONE" ]
then
	cat /tmp/dummy.$1.position.txt
	exit 0
fi

DEVICE=$(ticcmd --list|grep $1| cut -d, -f1)
if [ "$DEVICE" = "" ]
then
	if [ ! -f /tmp/last.motor.pos.$1 ]
	then
		echo 0 > /tmp/last.motor.pos.$1
	fi
	cat /tmp/last.motor.pos.$1
else
       POSITION=$(ticcmd -d $DEVICE -s --full| grep position| grep -i current | cut -d: -f2 | awk '{print $1}')
       echo "$POSITION" > /tmp/last.motor.pos.$1
       echo "$POSITION"
fi


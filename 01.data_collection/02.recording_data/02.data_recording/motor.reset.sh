#!/bin/bash

if [ $# -lt 1 ]
then
    echo "Usage: $0 <T500/T825>"
    exit 1
fi


DEVICE=$(ticcmd --list|grep $1| cut -d, -f1)

if [ "$DEVICE" == "" ]
then
    echo "$1 device not found !!"
    exit 1
fi

ticcmd -d $DEVICE --deenergize
ticcmd -d $DEVICE --reset

ticcmd -d $DEVICE --get-settings tic-$DEVICE-settings
perl -pi -e  's/command_timeout.*/command_timeout: 0/' tic-$DEVICE-settings
ticcmd -d $DEVICE --settings tic-$DEVICE-settings

ticcmd -d $DEVICE --halt-and-set-position 0




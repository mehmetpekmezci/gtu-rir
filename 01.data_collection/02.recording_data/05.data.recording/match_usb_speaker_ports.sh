#!/bin/bash



if [ $# -lt 1 ]
then
   echo "Usage $0 [ TEST_MODE_NONE | TEST_MODE_FAST_DEVICE_MOC | TEST_MODE_DEVICE_MOC]"
   exit 1
fi   


TEST_MODE=$1

if [ "$TEST_MODE" != "TEST_MODE_NONE" ]
then
   ## HERE IS TEST MODE
   SPEAKER_0=$(pacmd list-sinks | grep 'name:' |  cut -d\< -f2| cut -d\> -f1)
   echo "from header import * " > speaker_numbers.py
   echo "SPEAKERS=['$SPEAKER_0','$SPEAKER_0']" >>speaker_numbers.py
   echo "NUMBER_OF_SPEAKERS=2*len(SPEAKERS)" >>speaker_numbers.py

else 
   ## HERE IS REAL MODE
   pacmd list-sinks | grep  -i "name:\|long_card_name" | grep -i usb> /var/tmp/sinks
   perl -0pi -e 's/>\n/>/imsg' /var/tmp/sinks
   # MP : HERE 2.4 and 2.3 are last two digits of the usb port numbers of the speakers,
   #      i found them by ppluging/unplugging their cables
   SPEAKER_0=$(cat /var/tmp/sinks | grep '4,'| cut -d\< -f2| cut -d\> -f1)
   SPEAKER_1=$(cat /var/tmp/sinks | grep '3,'| cut -d\< -f2| cut -d\> -f1)


   echo "from header import * " > speaker_numbers.py
   echo "SPEAKERS=['$SPEAKER_0','$SPEAKER_1']" >>speaker_numbers.py
   echo "NUMBER_OF_SPEAKERS=2*len(SPEAKERS)" >>speaker_numbers.py

fi

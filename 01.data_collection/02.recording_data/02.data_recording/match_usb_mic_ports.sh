#!/bin/bash



if [ $# -lt 1 ]
then
   echo "Usage $0 [ TEST_MODE_NONE | TEST_MODE_FAST_DEVICE_MOC | TEST_MODE_DEVICE_MOC]"
   exit 1
fi   


TEST_MODE=$1

if [ "$TEST_MODE" != "TEST_MODE_NONE" ]
then
    echo "from header import * " > microphone_numbers.py
    
    for line in $(grep -v '#' usb_microphone.conf)
    do
        usb_port=$(echo $line| cut -d: -f1)
        mic_no=$(echo $line| cut -d: -f2)
        echo "usbport[$mic_no]=$mic_no" >> microphone_numbers.py
        echo "micport[$mic_no]=$mic_no" >> microphone_numbers.py
        echo "" >> microphone_numbers.py
    done
    
else

    cat /proc/asound/cards > /var/tmp/cardlist
    #cp var_tmp_cardlist /var/tmp/cardlist
    perl -0pi -e 's/\s*MUSIC/ MUSIC/imsg' /var/tmp/cardlist
    cat /var/tmp/cardlist | grep Microphone > /var/tmp/cardlist.1
    mv /var/tmp/cardlist.1 /var/tmp/cardlist
    #cat /var/tmp/cardlist
    cat /var/tmp/cardlist | sed -e 's/.*usb-/usb-/'| sed 's/,.*/,/'> /var/tmp/usblist
    USB_BASE_ADDR=$(cat /var/tmp/usblist| awk '{ print length, $0 }' | sort -s | cut -d" " -f2-|head -1| tr '.' '\n'|grep -v ,| tr '\n' '.')

    echo $USB_BASE_ADDR

    echo "from header import * " > microphone_numbers.py

    for line in $(grep -v '#' usb_microphone.conf)
    do
        usb_port=$(echo $line| cut -d: -f1)
        mic_no=$(echo $line| cut -d: -f2)
        N=$(cat /var/tmp/cardlist | grep "$USB_BASE_ADDR$usb_port" | awk '{print $1}')
        echo "#### USB PORT : $USB_BASE_ADDR$usb_port, ALSA MIC DEVICE NO : $N"
        echo "" >> microphone_numbers.py
        echo "usbport[$N]=$mic_no" >> microphone_numbers.py
        echo "micport[$mic_no]=$N" >> microphone_numbers.py
        echo "" >> microphone_numbers.py
    done

fi

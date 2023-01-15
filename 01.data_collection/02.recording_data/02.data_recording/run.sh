#!/bin/bash

rm -f /tmp/last.motor.pos.*
rm -f /tmp/dummy.*

if [ $# -lt 1 ]
then
   echo "Usage $0 [ TEST_MODE_NONE | TEST_MODE_FAST_DEVICE_MOC | TEST_MODE_DEVICE_MOC ]"
   exit 1
fi   


TEST_MODE=$1




echo "Date is : "
date

echo "If you want to configure the time use date command, example : 'date -s 12:34' , this command set the time og the machine to 12:34 "

echo "Do you want to continue ( Time is correct) (y/n) ?"
read ans
if [ "$ans" == "y" ]
then
	echo "Start rir sound measurement system "
else
	echo "Set your time using date -s command."
	exit 1
fi

pip3 list| grep lywsd03mmc
if [ $? != 0 ]
then
 #echo list | bluetoothctl
 sudo apt-get install python3-pip libglib2.0-dev
 sudo pip3 install bluepy
 sudo pip3 install lywsd03mmc
 sudo apt install python3-pip
 sudo pip3 install librosa
 sudo pip3 install numpy==1.21
 sudo apt install python3-pyaudio
 sudo pip3 install pyaudio
 sudo apt install python3-opencv
 sudo apt install ffmpeg
# sudo apt install pkg-config libflac-dev libogg-dev 
fi

./match_usb_mic_ports.sh $TEST_MODE
./match_usb_speaker_ports.sh $TEST_MODE 

#cat TODO_NOTES.txt
#sleep 2

#echo "REBUILT BLUEPY FROM SOURCE !!! it was blocking (deadlock)"
#sleep 2

sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target


./test_devices.sh $TEST_MODE
if [ $? != 0 ]
then
	echo "test_device.sh found some missing devices, please check your devices and run again"
	exit 1
fi

if [ ! -f usbreset ]
then
	gcc -o usbreset usbreset.c
	chmod +x usbreset
fi

#echo "Set of USB Devices :"
#echo "--------------------"
#lsusb | cut -d: -f1| awk '{print "/dev/bus/usb/"$2"/"$4}'
#echo
#echo
#lsusb -t
#echo
#echo
# TO RESET USB 
# sudo su -c 'echo -n "0000:00:1d.0" | tee /sys/bus/pci/drivers/ehci-pci/unbind'
# sudo su -c 'echo -n "0000:00:1d.0" | tee /sys/bus/pci/drivers/ehci-pci/bind'
# sudo su -c 'echo -n "0000:00:1a.0" | tee /sys/bus/pci/drivers/ehci-pci/unbind'
# sudo su -c 'echo -n "0000:00:1a.0" | tee /sys/bus/pci/drivers/ehci-pci/bind'


WHOAMI=$(whoami)

sudo grep $WHOAMI /etc/sudoers >/dev/null
if [ $? != 0 ]
then
    echo "Please insert the line  '$WHOAMI     ALL=(ALL) NOPASSWD:ALL'  to the END of file /etc/sudoers  "
    exit 1
fi

sudo grep "usbcore.autosuspend" /etc/default/grub > /dev/null
if [ $? != 0 ]
then
    echo "Please insert the line  'GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.autosuspend=-1"'  to the /etc/default/grub  "
    exit 1
fi

echo "Did you test the turning arms of mechanism? (Press Enter when done)"
read

echo "Did you isolated motor controller cards from vibration? (if not this coups usb connection temporarily)"
read

echo "Did you put the microphone and speaker stands GREEN STICKERS FACING THE DOOR OF THE ROOM (starting position)? (Press Enter when done)"
read

echo "Did you position the webcam to view the microphone and speaker stands? (Press Enter when done)"
read

USERNAME=$(whoami)
echo "sudo renice -n -20  -u $USERNAME"
sudo renice -n -20  -u $USERNAME
echo

## DONT REMOVE touch /tmp/tempHum.txt
touch /tmp/tempHum.txt

#echo "Record Numbers from microphone to guarantee microphone-place matching ..."
#echo "1-9 on the microphone stand:"
#python3 record_mic_numbers.py
#reset

#echo "10-11 on the speaker stand:"
#python3 record_mic_numbers.py
#reset

echo "Starting ..."

if [ "$TEST_MODE" = "TEST_MODE_NONE" ]
then
    ./get_temp_humidity_periodically.sh &
fi

if [ "$TEST_MODE" != "TEST_MODE_NONE" ]
then
	echo 0 > /tmp/dummy.T825.position.txt
	echo 0 > /tmp/dummy.T500.position.txt
fi

python3 main.py $TEST_MODE



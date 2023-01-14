#!/bin/bash
#lsusb -t


if [ $# -lt 1 ]
then
   echo "Usage $0 [ TEST_MODE_NONE | TEST_MODE_FAST_DEVICE_MOC | TEST_MODE_DEVICE_MOC]"
   exit 1
fi   


TEST_MODE=$1

if [ "$TEST_MODE" != "TEST_MODE_NONE" ]
then
   exit 0
fi   

./match_usb_mic_ports.sh $TEST_MODE
./match_usb_speaker_ports.sh $TEST_MODE


echo
echo "##########################################"
echo "############ CHECKING BLUETOOTH DEVICES"
echo "##########################################"
echo
NUMBER_OF_BLUETOOTH_DEVICES=$(rfkill list| grep -i hci| wc -l)
if [ $NUMBER_OF_BLUETOOTH_DEVICES -lt 1 ]
then
	echo "You miss the bluetooth device ..."
	echo "Number of devices = $NUMBER_OF_BLUETOOTH_DEVICES "
        echo "rfkill list| grep -i hci"
        rfkill list| grep -i hci
fi

rfkill list | grep -A3 -i bluetooth | grep blocked| grep yes  > /dev/null
if [ $? != 0 ]
then
   echo "Did you turn ON the BLUETOOTH device of your laptop? (Press Enter when done)"
   read
fi



echo
echo "##########################################"
echo "############ CHECKING USB MICROPHONES"
echo "##########################################"
echo


cat /proc/asound/cards > /var/tmp/cardlist
perl -0pi -e 's/\s*MUSIC/ MUSIC/imsg' /var/tmp/cardlist
cat /var/tmp/cardlist | grep Microphone > /var/tmp/cardlist.1
mv /var/tmp/cardlist.1 /var/tmp/cardlist
#cat /var/tmp/cardlist


NUMBER_OF_USB_MICROPHONES=$(cat /var/tmp/cardlist| wc -l)
if [ $NUMBER_OF_USB_MICROPHONES -lt 6 ]
then
	echo "You miss some of the usb microphones (min 6)..."
	echo "Number of usb microphones = $NUMBER_OF_USB_MICROPHONES "
        echo "arecord -L | grep hw:CARD| grep -v plughw"
	arecord -L | grep hw:CARD| grep -v plughw
        echo "cat /proc/asound/cards"
        cat /var/tmp/cardlist
	exit 1
fi

./match_usb_mic_ports.sh $TEST_MODE

echo "Do you want to check microphones one by one (y/n)? "
read ans
if [ "$ans" == "y" ]
then
echo "Starting Microphone Check, Press Enter When Ready .."
read

for((i=0;$i<6;i=$i+1))
do
   RECORDING_SLEEP_TIME=1
   RECORD_DURATION=3
   echo "Say $i to the microphone-$i (Press Enter To Start ...) :"
   read
   echo "Recording will start in $RECORDING_SLEEP_TIME :"
   sleep $RECORDING_SLEEP_TIME
   USBPORT=$(grep "micport\[$i\]" microphone_numbers.py| cut -d'=' -f2)
   rm -f /var/tmp/temp.wav
   ffmpeg  -f  alsa  -ac  1  -sample_rate 44100 -i hw:$USBPORT,0 -t $RECORD_DURATION -f s32le /var/tmp/temp.wav
   ffplay  -autoexit  -f  alsa  -ac  1  -sample_rate 44100  -f s32le -t $RECORD_DURATION  /var/tmp/temp.wav
   echo "Did you hear $i back (y/n)? "
   read ans
   if [ "$ans" == "n" ]
   then
          "Microphone $i can not be heard... quiting."
          exit 1
   fi
   rm -f /var/tmp/temp.wav
done
fi

echo "Check microphone directions (in case of movement) :"
echo "0: RIGHT"
echo "1: DOWN"
echo "2: OUT_TO_FACE"
echo "3: LEFT"
echo "4: DOWN"
echo "5: OUT_TO_FACE"
echo "Press Enter When Done"
read


echo
echo "##########################################"
echo "############ SETTING MICROPHONE VOLUMES TO 70000 (MAX 90000)"
echo "##########################################"
echo
for i in $(pacmd list-sources| grep index: | sed -e 's/.*index//' | awk '{print $2}')
do
    pacmd set-source-volume $i 70000
done




echo
echo "##########################################"
echo "############ CHECKING SPEAKERS "
echo "##########################################"
echo
NUMBER_OF_SPEAKERS=$(pacmd list-sinks | grep name: | grep -i usb| wc -l)
if [ $NUMBER_OF_SPEAKERS -lt 2 ]
then
	echo "You miss some of the spkeakers (min 2)..."
	echo "Number of speakers = $NUMBER_OF_SPEAKERS "
        echo "pacmd list-sinks | grep name:| grep -i usb"
	pacmd list-sinks | grep name:| grep -i usb
fi

echo
echo "##########################################"
echo "############ SETTING SPEAKER VOLUMES TO 90000 (MAX)"
echo "##########################################"
echo
for i in $(pacmd list-sinks| grep index: | sed -e 's/.*index//' | awk '{print $2}')
do
    pacmd set-sink-volume $i 70000 # MAX 90000
done

python3 test_speakers.py

echo
echo "##########################################"
echo "############ CHECKING STEP MOTORS"
echo "##########################################"
echo
NUMBER_OF_MOTORS=$(ticcmd --list| wc -l)
if [ $NUMBER_OF_MOTORS -lt 2 ]
then
	echo "You miss some of the motors (min 2)..."
	echo "Number of motors = $NUMBER_OF_MOTORS "
        echo "ticcmd --list"
	ticcmd --list
fi


echo
echo "##########################################"
echo "############ CHECKING WEBCAMS"
echo "##########################################"
echo
NUMBER_OF_WEBCAMS=$(lsusb| grep -i "microdia"| wc -l)
if [ $NUMBER_OF_WEBCAMS -lt 1 ]
then
	echo "You miss external webcam ..."
	echo "Number of webcams = $NUMBER_OF_WEBCAMS "
        echo "lsusb| grep -i "microdia"| wc -l"
        lsusb| grep -i "microdia"| wc -l
fi

./test_webcam.sh

#echo
#echo "##########################################"
#echo "############ CHECKING TEMPERATURE SENSORS"
#echo "##########################################"
#echo
#python3 test_temp_hum_sensor.py
#
#if [ $? != 0 ]
#then
#	echo "Cannot reach temperature/humdity sensors by bluetooth"
#        echo "python3 test_temp_hum_sensor.py"
#fi


echo
echo
echo

if [ $NUMBER_OF_BLUETOOTH_DEVICES -lt 1 ]
then 
	exit 1
fi

if [ $NUMBER_OF_USB_MICROPHONES -lt 6 ]
then
	exit 1
fi

if [ $NUMBER_OF_SPEAKERS -lt 2 ]
then
    if [ "$NO_CHECK" == "1" ]
    then
	echo exit 1
    else
	exit 1
    fi
fi

if [ $NUMBER_OF_MOTORS -lt 2 ]
then
    if [ "$NO_CHECK" == "1" ]
    then
	echo exit 1
    else
	exit 1
    fi
fi

if [ $NUMBER_OF_WEBCAMS -lt 1 ]
then
	exit 1
fi

#sudo su -c 'echo -n "0000:00:1d.0" | tee /sys/bus/pci/drivers/ehci-pci/unbind'
#sleep 15
#sudo su -c 'echo -n "0000:00:1d.0" | tee /sys/bus/pci/drivers/ehci-pci/bind'
#sleep 15
#sudo su -c 'echo -n "0000:00:1a.0" | tee /sys/bus/pci/drivers/ehci-pci/unbind'
#sleep 15
#sudo su -c 'echo -n "0000:00:1a.0" | tee /sys/bus/pci/drivers/ehci-pci/bind'
#sleep 15
#
#sleep 20


#!/bin/bash

lsusb -t| grep -i 'usbhid\|hub'

cat /proc/asound/cards > /var/tmp/cardlist
perl -0pi -e 's/\s*MUSIC/ MUSIC/imsg' /var/tmp/cardlist

cat /var/tmp/cardlist | grep Microphone > /var/tmp/cardlist.1

mv /var/tmp/cardlist.1 /var/tmp/cardlist

echo "usb_port:"
cat /var/tmp/cardlist 

echo "mic_no is the port number label on the usb hub hub"

### I commented out the exit line below, 
### and rerun this script untill determining all the usb adresses of microphone devices
### by unplugging and plugging the usb devices on the usb hub.
#exit 0


echo '
#usb_port:mic_no
4:0
3:1
1.1:2
1.4:3
1.3:4
1.2:5
' > usb_microphone.conf.test












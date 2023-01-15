#!/bin/bash

# lspci -Dk| grep -i "USB controller:"
####### >>>>>>    buradaki pci adresini

# lspci -Dk
###  Kernel driver karisiliginda xhci_hcd veya ehci-pci .... olabilir



DEVICE_TEST_FAILED=1

while [ $DEVICE_TEST_FAILED = 1 ]
do
#   if [ $DEVICE_TEST_FAILED = 1 ]
#   then
	echo
	echo "###############################################################"
	echo "############ RESETTING USB PORTS "
	echo "###############################################################"
	echo

	sudo su -c 'echo -n "0000:00:14.0" > /sys/bus/pci/drivers/xhci_hcd/unbind'
	sleep 15
	sudo su -c 'echo -n "0000:00:14.0" > /sys/bus/pci/drivers/xhci_hcd/bind'
	sleep 15


	lsusb
	lsusb -t
        sleep 30
#    fi
   /home/rir/workspace-python/room_impulse_response_phd_thessis/src/rir-measurement/05.data.recording/test_devices.sh
   if [ $? = 0 ]
   then
     DEVICE_TEST_FAILED=0
   else
     DEVICE_TEST_FAILED=1
   fi
done



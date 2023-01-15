#!/bin/bash

while true
do
sudo su -c "kill -9 \$(ps -ef | grep -v grep | grep bluepy-helper| awk '{print \$2}')"
echo "sudo hciconfig hci0 down"
sudo hciconfig hci0 down
echo "sudo hciconfig hci0 up"
sudo hciconfig hci0 up
sleep 7200
#sleep 2 hours
done


hcitool scan  # to get the MAC address of your device
bluetoothctl
agent on
scan on  # wait for your device's address to show up here
scan off
trust MAC_ADDRESS
pair MAC_ADDRRESS
connect MAC_ADDRESS



rfkill block bluetooth
rfkill unblock bluetooth
sudo hciconfig hci0 noauth
sudo hciconfig hci1 noauth
sudo hciconfig hci2 noauth

hciconfig -a


for i in $(hciconfig -a| grep  hci.:| cut -d: -f1)
do
        hciconfig $i| grep '00:15:83:0C:BF:EB' > /dev/null
        if [ $? = 0 ]
        then
                echo "$i-00:15:83:0C:BF:EB"
        fi

        hciconfig $i| grep '68:07:15:D4:9E:E5' > /dev/null
        if [ $? = 0 ]
        then
                echo "$i-68:07:15:D4:9E:E5"
        fi

done



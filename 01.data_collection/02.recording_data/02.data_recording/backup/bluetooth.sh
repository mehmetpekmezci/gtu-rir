 apt install bluez bluez-* pavucontrol pulseaudio-module-bluetooth
 sudo apt-get install blueman -y && blueman-manager
rfkill list
python3 bluetooth_temperature_humidity_xiaomi.py

Ok, I have discovered that it is possible and I discovered how:

    I installed all bluez (bluez + bluez-*) packages and purged any other application related to bluetooth (blueman, bluewho, etc.). I do not know if this is strictly relevant, but until I didn't do this I couldn't manage to connect to the speakers.

    I also installed all the pulseaudio utilities and configured the simultaneous output virtual device.

    Connect your 2 dongles. They will get the names hci0 and hci1

    You should check that the devices are not blocked with:

    rfkill list

    If you find any blockage on the bluetooth interfaces (this command will also show your wifi) you need to unblock it (check rfkill man page to proceed).

    Check that there are no devices paired to your bluetooth interfaces with:

    bt-device -a hci0 -l

    bt-device -a hci1 -l

    If there are paired devices I preferred to delete all previous pairings before continuing with:

    bt-device -a hciX -r XX:XX:XX:XX:XX:XX

    Check that your devices can be reached from the dongles by discovering them with:

    hcitool -i hci0 scan

    hcitool -i hci1 scan

    With the previous step you will get the bluetooth mac addresses of the devices (the string that looks like XX:XX:XX:XX:XX:XX). With those numbers you should pair the speakers with (I added a 1 and 2 at the end of the mac addresses to identify the 2 different speakers):

    bt-device -a hci0 -c XX:XX:XX:XX:XX:X1

    bt-device -a hci1 -c XX:XX:XX:XX:XX:X2

    Connect to the speakers for audio with:

    bt-audio -a hci0 -c XX:XX:XX:XX:XX:X1

    bt-audio -a hci1 -c XX:XX:XX:XX:XX:X2

echo "help" | bluetothctl
echo "list" | bluetothctl
echo "scan on" | bluetothctl

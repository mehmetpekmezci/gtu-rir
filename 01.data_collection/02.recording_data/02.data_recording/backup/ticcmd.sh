echo "install ticcmd from ../../driver/pololu-tic-1.8.1-linux-x86.tar.xz if not already installed"
ticcmd --list
TIC_T500=$(ticcmd --list|grep T825| cut -d, -f1)
ticcmd --reset

ticcmd -d $TIC_T500 --get-settings tic-t825-settings
perl -pi -e  's/command_timeout.*/command_timeout: 0/' tic-t825-settings
ticcmd -d $TIC_T500 --settings tic-t500-settings

#ticcmd --resume --exit-safe-start --position 200
#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity 2000000 --position -2000000  &
ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity 2000000 --position 0  &
sleep 5
#ticcmd --halt-and-set-position 0
ticcmd --deenergize
#ticcmd --reset

#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity -2000000 --position -2000000  &
ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity -2000000 --position 0  &
sleep 5
#ticcmd --halt-and-set-position 0
ticcmd --deenergize
#ticcmd --reset


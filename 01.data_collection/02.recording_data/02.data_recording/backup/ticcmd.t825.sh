echo "install ticcmd from ../../driver/pololu-tic-1.8.1-linux-x86.tar.xz if not already installed"
ticcmd --list
TIC_T825=$(ticcmd --list|grep T825| cut -d, -f1)
ticcmd -d  $TIC_T825 --reset

ticcmd -d $TIC_T825 --get-settings tic-t825-settings
perl -pi -e  's/command_timeout.*/command_timeout: 0/' tic-t825-settings
ticcmd -d $TIC_T825 --settings tic-t825-settings
#ticcmd --resume --exit-safe-start --position 200
#ticcmd -d $TIC_T825 --resume --exit-safe-start --velocity 2000000 --position -2000000  &
#ticcmd --halt-and-set-position 0
#ticcmd --reset


#STEP=4
STEP=8
SLEEP=2

for((i=0;$i<$STEP;i=i+1))
do
        ticcmd --resume --exit-safe-start --position $((40*i))
sleep $SLEEP
ticcmd -d $TIC_T500 --deenergize
done

ticcmd -d  $TIC_T825 --reset

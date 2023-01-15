echo "install ticcmd from ../../driver/pololu-tic-1.8.1-linux-x86.tar.xz if not already installed"
ticcmd --list
TIC_T500=$(ticcmd --list|grep T500| cut -d, -f1)
ticcmd -d $TIC_T500 --reset

ticcmd -d $TIC_T500 --get-settings tic-t500-settings
perl -pi -e  's/command_timeout.*/command_timeout: 0/' tic-t500-settings
ticcmd -d $TIC_T500 --settings tic-t500-settings


#STEP=7
STEP=10
SLEEP=2

ticcmd --halt-and-set-position 0

for((i=0;$i<$STEP;i=i+1))
do

#ticcmd --resume --exit-safe-start --position 200
#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity 2000000 --position -2000000  &
#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity 30000 --position 0  &
ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity 100000  &
sleep $SLEEP
ticcmd -d $TIC_T500 --halt-and-hold
ticcmd -d $TIC_T500 -s --full| grep position
sleep 5
done

ticcmd -d $TIC_T500 --reset
ticcmd -d $TIC_T500 --deenergize
sleep 5
exit 0


for((i=0;$i<$STEP;i=i+1))
do
#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity -2000000 --position -2000000  &
#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity -30000 --position 0  &
#ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity -200000  &
ticcmd -d $TIC_T500 --resume --exit-safe-start --velocity -100000  &
sleep $SLEEP
#ticcmd --halt-and-set-position 0
#ticcmd -d $TIC_T500 --deenergize
ticcmd -d $TIC_T500 --halt-and-hold
ticcmd -d $TIC_T500 -s --full| grep position
sleep 5
#ticcmd --reset
done

ticcmd -d $TIC_T500 --reset
ticcmd -d $TIC_T500 --deenergize

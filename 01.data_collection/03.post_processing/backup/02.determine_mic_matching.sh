#!/bin/bash



CURRENT_DIR=$(pwd)

DATA_DIR=$(realpath "../../../data/single-speaker");
SCRIPT_DIR=$(pwd)
MIC_DATA_DIR=$(realpath "../../../data/single-speaker-mics/");
if [ ! -e $MIC_DATA_DIR ]
then
	mkdir -p $MIC_DATA_DIR
fi

N=$(find $MIC_DATA_DIR -name mic_plot.png | wc -l)
if [ $N -eq 0 ]
then

cd $DATA_DIR

for i in *mic*; do   last_date=$(ls $i| tail -1); cp -Rf $i/$last_date/iteration-00/speaker-iteration-0/microphone-iteration-0/active-speaker-0/channelNo-0/ $MIC_DATA_DIR/$i ;done

cd $MIC_DATA_DIR
find . -name record.txt | xargs rm -f
find . -name *.jpg | xargs rm -f
find . -name transmit* | xargs rm -f
for i in $(find . -name *.tar.bz2); do  echo $i; d=$(dirname $i);   tar -xvjf  $i -C $d; rm $i; done
for i in $(find . -name *.bz2); do  bunzip2  $i ; done

for i in *
do
        python3 $SCRIPT_DIR/mic_plot.py $i
done

for i in *
do
	room_config=$(echo $i | sed -e 's/.mic.*//')
	touch $room_config.mics.txt
done


for i in *.mics.txt
do
echo '
1:
2:
3:
4:
5:
6:
7:
8:
9:
10:
11:
' > $i
gedit $i &
done

fi

cd $MIC_DATA_DIR

for i in $(ls * -d| grep -v txt)
do  
	eog $i/mic_plot.png & 
	room_config=$(echo $i | sed -e 's/.mic.*//')
	gedit $room_config.mics.txt & 
	read
done


#echo " LISTEN TO THE WAV FILES AND DETERMINE MICROPHONE MATCHINGS FOR EACH CONFIGURATION"
#echo " Format will be like below, in each configuration file <room_name>.mics.txt"
#echo " Real Mic Placing : Record Number "
#echo " 1:5 "
#echo " 2:7 "
#
#read

#nautilus $MIC_DATA_DIR





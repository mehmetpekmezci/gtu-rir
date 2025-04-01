#!/bin/bash

GTU_RIR_DATA_DIR=$1

cp $GTU_RIR_DATA_DIR/transmittedSongSignal.wav.bz2 .

bunzip2 transmittedSongSignal.wav.bz2

python3 plot.py transmittedSongSignal.wav


if [ 1 = 1 ]
then

for i in $(find . -name '[1-4]-*.wav' | grep -v reverbed) 
do

   (
   echo python3 convolve.py transmittedSongSignal.wav $i
   python3 convolve.py transmittedSongSignal.wav $i
#   )&
   )
done
wait




for i in $(find . -name 'real.rir.*.wav' | grep -v reverbed) 
do
   (
   echo python3 convolve.py transmittedSongSignal.wav $i
   python3 convolve.py transmittedSongSignal.wav $i
#   )&
   )
done
wait


for i in $(find . -name 'real.rir.*.wav' | grep -v reverbed) 
do
   (
   echo python3 plot.py $i
   python3 plot.py $i
#   )&
   )
done
wait


fi


for i in $(find . -name real.song.*.wav ) 
do
   (
   echo python3 plot.py  transmittedSongSignal.wav $i
   python3 plot.py transmittedSongSignal.wav $i
#   )&
   )
done
wait


## plot 0
python3 plot.py 


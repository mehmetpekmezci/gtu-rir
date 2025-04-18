#rm -Rf data
mkdir -p data
mkdir -p data/real-song
mkdir -p data/real-rir

if [ ! -f data/real-song/room-207*.wav ]
then
for i in $(cat combinations.txt); do   if [ ! -f ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$i ] ; then bunzip2 ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$i.bz2 ; fi; done
for i in $(cat combinations.txt); do   ls ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$i; done
for i in $(cat combinations.txt); do   j=$(echo $i| sed -e 's/receivedEssSignal/receivedSongSignal/' | sed -e 's/wav.ir.//'); filename=$(echo $j| sed -e 's#/micx-./#-#'| sed -e 's#/receivedSongSignal#-micno#'|sed -e 's/wav.ir/rir/'); bunzip2 ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$j.bz2; ls ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$j; echo $filename; cp -f ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$j data/real-song/$filename; done
for i in $(cat combinations.txt); do   filename=$(echo $i| sed -e 's#/micx-./#-#'| sed -e 's#/receivedEssSignal#-micno#'|sed -e 's/wav.ir/rir/'); cp -f ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$i data/real-rir/$filename; done
fi

combinations=$(cat combinations.txt| sed -e 's#/micx-./micstep##'| sed -e 's#/receivedEssSignal##'| sed -e 's#room-##'|sed -e 's#spkstep-##'| sed -e 's#spkno-##'| sed -e 's#\..*##' | tr '\n' ' ')


GENERATED_RIR_HOME=~/RIR_REPORT/GTURIR/
if [ -d $GENERATED_RIR_HOME ]
then
for model in $(ls $GENERATED_RIR_HOME| grep MESHTAE)
do
    mkdir -p data/$model/
    for combination in $combinations
    do
	    ROOM_ID=$(echo $combination | cut -d- -f1)
	    MIC_ITR=$(echo $combination | cut -d- -f2)
	    SPK_ITR=$(echo $combination | cut -d- -f3)
	    SPK_NO=$(echo $combination | cut -d- -f4)
	    MIC_NO=$(echo $combination | cut -d- -f5)
            cp -f $GENERATED_RIR_HOME/$model/room-$ROOM_ID/mic*/SPEAKER_ITERATION-$SPK_ITR-MICROPHONE_ITERATION-$MIC_ITR-PHYSICAL_SPEAKER_NO-$SPK_NO-MICROPHONE_NO-$MIC_NO.wav data/$model/room-$ROOM_ID-SPEAKER_ITERATION-$SPK_ITR-MICROPHONE_ITERATION-$MIC_ITR-PHYSICAL_SPEAKER_NO-$SPK_NO-MICROPHONE_NO-$MIC_NO.rir.wav 1>/dev/null
            cp -f $GENERATED_RIR_HOME/$model/room-$ROOM_ID/mic*/SPEAKER_ITERATION-$SPK_ITR-MICROPHONE_ITERATION-$MIC_ITR-PHYSICAL_SPEAKER_NO-$SPK_NO-MICROPHONE_NO-$MIC_NO.wave.png data/$model/room-$ROOM_ID-SPEAKER_ITERATION-$SPK_ITR-MICROPHONE_ITERATION-$MIC_ITR-PHYSICAL_SPEAKER_NO-$SPK_NO-MICROPHONE_NO-$MIC_NO.rir.coherence.plot.png 1>/dev/null
    done
done
else
	echo "$GENERATED_RIR_HOME does not exists!"
fi

#rm -Rf data

combinations=$(cat combinations.txt| sed -e 's#/micx-./micstep##'| sed -e 's#/receivedEssSignal##'| sed -e 's#room-##'|sed -e 's#spkstep-##'| sed -e 's#spkno-##'| sed -e 's#\..*##' | tr '\n' ' ')
#python3 extract_mic_spk_coordinates.py ~/RIR_DATA/GTU-RIR/RIR.pickle.dat $combinatons

for model in $(ls  data| grep -v real| grep -v referenceSong.wav )
do
    for combination in $combinations
    do
            echo $model-$combination
	    ROOM_ID=$(echo $combination | cut -d- -f1)
	    MIC_ITR=$(echo $combination | cut -d- -f2)
	    SPK_ITR=$(echo $combination | cut -d- -f3)
	    SPK_NO=$(echo $combination | cut -d- -f4)
	    MIC_NO=$(echo $combination | cut -d- -f5)
            python3 convolve.py data/referenceSong.wav data/$model/room-$ROOM_ID-SPEAKER_ITERATION-$SPK_ITR-MICROPHONE_ITERATION-$MIC_ITR-PHYSICAL_SPEAKER_NO-$SPK_NO-MICROPHONE_NO-$MIC_NO.rir.wav 
    done
done

for realrir in  data/real-rir/*.rir.wav
do
        echo $realrir
	python3 convolve.py data/referenceSong.wav $realrir
done


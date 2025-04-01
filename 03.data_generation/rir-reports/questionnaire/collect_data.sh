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


#combinations=$(cat combinations.txt| sed -e 's#/micx-./micstep##'| sed -e 's#/receivedEssSignal##'| sed -e 's#room-##'|sed -e 's#spkstep-##'| sed -e 's#spkno-##'| sed -e 's#\..*##' | tr '\n' ' ')
#python3 extract_mic_spk_coordinates.py ~/RIR_DATA/GTU-RIR/RIR.pickle.dat $combinatons





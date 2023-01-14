MIC_DATA_DIR=$(realpath "../../../data/single-speaker-mics/");

cd $MIC_DATA_DIR

for i in */soundcards.txt
do
perl -0pi -e 's/\n\s*MUSIC-BOOST/MUSIC-BOOST/mg' $i
perl -0pi -e 's/\n\s*C-Media/C-Media/mg' $i
perl -0pi -e 's/\n\s*HDA Intel/HDA Intel/mg' $i
done



rm -f *.addr.txt
rm  -f mic_addr_mapping.txt


for mics_mapping_file in *.mics.txt
do
   conf=$(echo $mics_mapping_file| sed -e 's/.mics.txt//')
   echo $conf
   for micNo in $(seq 11)
   do
       n=$(grep "^$micNo:." $mics_mapping_file| cut -d: -f2)  
       if [ "$n" != "10" -a "$n" != "11" ]
       then
	       n=" $n"
       fi
       addr=$(grep "$n \[" $conf*/soundcards.txt| sed -e 's/.*usb-.*:.*://'| sed -e 's/,.*//' | sort | uniq | tr '\n' '#') 
       echo "$micNo=$addr" >> $mics_mapping_file.addr.txt
   done
done

for micNo in $(seq 11); do  echo -n ">>>>>> $micNo = " >> mic_addr_mapping.txt;  grep "^$micNo=" *.mics.txt.addr.txt| cut -d\= -f2| sort| uniq | grep -v '#.*#' | tr '\n' ' ' >> mic_addr_mapping.txt;echo >> mic_addr_mapping.txt; done


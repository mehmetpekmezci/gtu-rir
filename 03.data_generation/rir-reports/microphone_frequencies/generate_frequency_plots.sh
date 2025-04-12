#!/bin/bash
mkdir -p data
MULTI_PLOT_ARGUMENTS=""
for wavfile in $(cat combinations.txt)
do   
	basewavfile=$(basename $wavfile)
	if [ ! -f data/$basewavfile ] 
	then 
		cp -f ~/RIR_DATA/GTU-RIR-1.0/data/single-speaker/$wavfile.bz2 data/
		bunzip2 data/$basewavfile.bz2 
        fi
	micNo=$(echo $wavfile| awk -F'-' '{print $NF}' | cut -d. -f1)
        MULTI_PLOT_ARGUMENTS="$MULTI_PLOT_ARGUMENTS data/$basewavfile $micNo"
done
echo python3 generate_frequency_plot.py $MULTI_PLOT_ARGUMENTS
python3 generate_frequency_plot.py $MULTI_PLOT_ARGUMENTS
rm -Rf data


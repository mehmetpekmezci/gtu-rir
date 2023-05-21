#!/bin/bash

find . -name '*.wav*'| xargs rm -f

./findMatchingRecords.sh  /data.ext4/mpekmezci/REPORTS ../../../data/single-speaker /data.ext4/mpekmezci/BUT_ReverbDB_rel_19_06_RIR-Only

./generateSampleSongs.sh ../../../data/single-speaker

./generateHtml.sh

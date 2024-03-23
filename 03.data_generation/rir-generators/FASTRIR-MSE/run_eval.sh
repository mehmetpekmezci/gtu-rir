#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Usage $0 <FAST_RIR_EVALUATION_DATA_PICKLE_FILE>  <GENERATED_RIRS_DIR>"
	exit 1
fi

FAST_RIR_EVALUATION_DATA_PICKLE_FILE=$1
GENERATED_RIRS_DIR=$2

if [ ! -f "$FAST_RIR_EVALUATION_DATA_PICKLE_FILE" ]
then
        echo "Pickle file $FAST_RIR_EVALUATION_DATA_PICKLE_FILE does not exist ..."
        exit 1
fi

rm -Rf $GENERATED_RIRS_DIR
mkdir -p $GENERATED_RIRS_DIR

ls generate/*pth* > /dev/null
if [ $? != 0 ]
then
	echo "There is no *.pth file in the generate directory, copy the netG_epoch_*.pth file you trained."
	echo "This file may be found as output/*/Model_RT/*.pth after training the model "
	 exit 1
fi

cd generate
./merge.sh
cd ../code_new

python3 main.py --cfg cfg/RIR_eval.yml --gpu 0 --eval_dir $FAST_RIR_EVALUATION_DATA_PICKLE_FILE --eval_target_dir $GENERATED_RIRS_DIR


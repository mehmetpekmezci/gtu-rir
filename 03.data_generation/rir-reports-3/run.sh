CURRENT_DIR=$(pwd)

export RIR_GENERATORS_DIR="../../rir-generators-4"

echo "setting RIR_GENERATORS_DIR=$RIR_GENERATORS_DIR"

sleep 10

for i in $(echo BUTReverbDB GTURIR)
do
	cd $CURRENT_DIR/$i
	if [ -f stop ]
        then
	    echo "Stopping run.sh "
	    exit 1
	fi
	echo "Running run.sh in directory $CURRENT_DIR/$i"
	./run.sh
done

cd $CURRENT_DIR/summary

./prepare_summary_html.sh

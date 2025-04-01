CURRENT_DIR=$(pwd)
for i in $(echo BUTReverbDB GTURIR)
do
	echo "Running run.sh in directory $CURRENT_DIR/$i"
	cd $CURRENT_DIR/$i
	./run.sh
done

cd $CURRENT_DIR/summary

./prepare_summary_html.sh

cd $CURRENT_DIR/questionnaire

./prepare_questionnaire.sh




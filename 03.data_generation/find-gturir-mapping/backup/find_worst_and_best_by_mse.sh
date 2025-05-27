RIR_REPORT_DIR=~/RIR_REPORT_BACKUP/RIR_REPORT.EVAL.MORE.NODE.AND.MORE.HEAD/GTURIR

CURRENT_DIR=$(pwd)
cd $RIR_REPORT_DIR
(
for i in MESHTAE.node.*/room-*/mic*/MSE*
do   
	echo $i | grep MESHTAE.node.2500>/dev/null; 
	if [ $? != 0 ]
	then 
		grep '0\.13' $i; 
		if [ $? = 0 ] 
		then 
			echo $i; 
		fi
	fi
done
) > $CURRENT_DIR/worst_record.txt

record=$(cat  $CURRENT_DIR/worst_record.txt |head -1| cut -d= -f1)
subdir=$(cat  $CURRENT_DIR/worst_record.txt |head -2 | tail -1 | sed -e 's#/MSE.*##')
model=$(echo $subdir| cut -d\/ -f1)
room=$(echo $subdir| cut -d\/ -f2)
for i in  $model/METADATA_*/Meshes/*$room*
do
	f=$(basename $i)
	cp $i $CURRENT_DIR/$f.worst
done
cp $model/METADATA_*/Meshes/loss* $CURRENT_DIR/loss.mesh.worst.txt
cp $subdir/$record*png  $CURRENT_DIR/


(
for i in MESHTAE.node.*/room-*/mic*/MSE*
do   
	echo $i | grep MESHTAE.node.2500>/dev/null; 
	if [ $? != 0 ]
	then 
		grep '0\.0024' $i; 
		if [ $? = 0 ] 
		then 
			echo $i; 
		fi
	fi
done
) > $CURRENT_DIR/best_record.txt
record=$(cat  $CURRENT_DIR/best_record.txt |head -1| cut -d= -f1)
subdir=$(cat  $CURRENT_DIR/best_record.txt |head -2| tail -1 | sed -e 's#/MSE.*##')
model=$(echo $subdir| cut -d\/ -f1)
room=$(echo $subdir| cut -d\/ -f2)
for i in  $model/METADATA_*/Meshes/*$room*
do
	f=$(basename $i)
	cp $i $CURRENT_DIR/$f.best
done
cp $model/METADATA_*/Meshes/loss* $CURRENT_DIR/loss.mesh.best.txt
cp $subdir/$record*png  $CURRENT_DIR/

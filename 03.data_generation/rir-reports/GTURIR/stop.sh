for  i in $(ps -ef | grep 'python3 mainData.py' | grep 'RIR_DATA/GTU-RIR-1.0'| awk '{print $2}')
do  
	kill -9 $i
done

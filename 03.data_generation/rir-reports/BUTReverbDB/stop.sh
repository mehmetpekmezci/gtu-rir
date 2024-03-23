for  i in $(ps -ef | grep 'python3 mainData.py' | grep 'RIR_DATA/BUTReverbDB'| awk '{print $2}')
do  
	kill -9 $i
done

